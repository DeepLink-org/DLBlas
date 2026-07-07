# engram_hash AscendC 算子开发计划

> **版本**: v1.0
> **日期**: 2026-07-01
> **关联设计**: [DESIGN.md](./DESIGN.md)
> **目标芯片**: Ascend 910B2 (DAV_2201) / CANN 9.0.0
> **技术路线**: AIV-only 标量整数计算（Vector 核，非 Cube，非 Vector 向量化）

---

## 1. 需求概述

| 项目 | 说明 |
|------|------|
| 算子名称 | engram_hash |
| 功能 | N-gram embedding 索引哈希：mul(int64) → XOR 前缀链 → mod → +offset |
| 输入 | ngram_token_ids(int32 [NT,N]), multipliers(int64 [L,N]), vocab_sizes(int32 [L,N-1,T]), offsets(int32 [L,W]) |
| 输出 | output(int32 [L, NT, W])，W=(N-1)*T |
| 精度标准 | **整数计算类：bit-exact（二进制一致 / 绝对误差 0）** |
| 基线 shape | NT=4096, N=3, L=2, T=8 → 输出 (2,4096,16) |

### 1.1 核心约束（贯穿全程）

1. **bit-exact**：全 int64 标量精确运算，禁止浮点近似。已核算：product ≤34bit 无溢出、XOR 恒非负、mod 两操作数非负 ⇒ C++ `%` == PyTorch `%`。
2. **无矩阵乘 / 无浮点**：不使用 Cube；不使用 Vector int64（DAV_2201 不支持）。
3. **多 shape 通用**：kernel 参数化，不硬编码 NT/N/L/T。
4. **核数不硬编码**：`GetBlockNum()` / 平台接口获取，保证 910B3 可移植。

---

## 2. 实施步骤

### Phase 0: 环境与骨架（预计 0.5 天）

- [ ] 确认 CANN 9.0.0 可 source（`set_env.sh`），`ascendc --npu-arch=dav-2201` 可用
- [ ] 确认 Ascend910B2 设备可用（优先 NPU 2；避开 npu-smi 标 Alarm 的 4/6 号卡）
- [ ] 以 `test/MTPBlock` 为模板，拷贝目录骨架（op_kernel / op_host / op_extension / scripts / CMakeLists.txt）
- [ ] 定义 `engram_hash_tiling.h`：`EngramHashTilingData` 结构体 + `EH_UB_AVAIL` 常量

**里程碑 M0**：目录骨架就绪，CMake 可配置（空 kernel 能编译链接）。

### Phase 1: Golden 与数据（预计 0.5 天）

- [ ] `scripts/gen_data.py`：复用 `origin/engram_hash.py` 的 `generate_test_data` / `make_offsets` 生成 4 输入 + 调用 `Model.forward` 生成 golden，落盘二进制（int32/int64 分别存）
- [ ] `scripts/verify_result.py`：读 kernel 输出与 golden，按整数 bit-exact 判定（`np.array_equal`）
- [ ] 用 tiny case（如 NT=3,N=3,L=2,T=3）在纯 CPU 侧再次确认标量公式 == `Model.forward`（已初步验证，纳入脚本回归）

**里程碑 M1**：golden 生成 + bit-exact 校验脚本就绪。

### Phase 2: Kernel 核心实现（预计 1.5 天）

- [ ] `engram_hash_kernel.asc`：
  - [ ] `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + BlockIdx/blockNum 边界
  - [ ] 常驻表载入 UB（mult/vocab/offset）；multipliers int64 处理（首选 UB，兜底 GM 直读）
  - [ ] token tile 循环 + ngram tile DataCopy 入 UB
  - [ ] 标量三重循环（layer → token → i），内层 XOR 链 + T 次 mod+offset，写 outUb
  - [ ] 按 layer 分段 DataCopy 写出 output
  - [ ] EnQue/DeQue 同步（VECIN ngUb / VECOUT outUb）
- [ ] `engram_hash_host.asc`：命令行驱动（NT N L T [dev] [core]）→ 随机输入 → ComputeTiling → 启核 → D2H → 落盘
- [ ] ComputeTiling 实现（多核切分 + tileTokens 预算 + 对齐标志）

**里程碑 M2**：单核功能跑通，基线 shape 输出非空。

### Phase 3: 精度验证与边界（预计 1 天）

- [ ] 基线 shape bit-exact 通过（NT=4096,N=3,L=2,T=8）
- [ ] 多核（48 核）正确性：满核 + 尾核 token 边界
- [ ] 对齐边界：ngram tile / 输出段非 32B 对齐时用 DataCopyPad 验证
- [ ] 多 shape 遍历验证（见 §3 测试矩阵），全部 bit-exact
- [ ] 边界：NT<核数、N=2（W 最小）、T=1、大 NT 触发多 tile

**里程碑 M3**：全测试矩阵 bit-exact 通过（含多核、多 shape、边界）。

### Phase 4: PyTorch 集成（预计 0.5 天）

- [ ] `engram_hash_torch.cpp`：前端校验 + shape 推导 + ComputeTiling + kernel launch + 返回 output
- [ ] Meta 实现：输出 `(L,NT,W)` int32 形状推导
- [ ] `register.cpp`：`TORCH_LIBRARY`(npu) + PrivateUse1 + Meta
- [ ] CMake Target 2：`libengram_hash_ops.so`
- [ ] Python 侧 `torch.ops.npu.engram_hash(...)` 调用，与 `Model.forward` 对比 bit-exact

**里程碑 M4**：`torch.ops.npu.engram_hash` 可用且与参考一致。

### Phase 5: 性能优化（预计 1 天，迭代）

- [ ] `scripts/benchmark.py`：msprof / profiler 采集延迟
- [ ] P0：确认满核 token 切分 + 常驻表复用生效（对比欠核/满核加速比）
- [ ] P1：Double Buffer（ngUb/outUb depth=2）实测是否有收益（访存占比低，作为可选）
- [ ] P2：内层 T/N 循环展开（编译期已知时）减少标量控制开销
- [ ] 每次仅改一处，改后立即 bit-exact 复验，性能不劣化才保留

**里程碑 M5**：满核加速比达标（≥ 0.7×usedCores 相对单核），bit-exact 保持。

---

## 3. 测试计划

### 3.1 测试矩阵

| 维度 | 取值 | 说明 |
|------|------|------|
| num_tokens (NT) | 32, 256, 1024, 4096, 65536 | 欠核 / 满核 / 多 tile |
| max_ngram_size (N) | 2, 3, 4, 5 | XOR 链长度、P=N-1 |
| num_ngram_layers (L) | 1, 2, 4 | 核内 layer 循环 |
| num_tables (T) | 1, 4, 8, 16 | 输出宽度 W、mod 次数 |
| 核数 | 1, 8, 24, 48 | 并行度 + 尾核边界 |

### 3.2 冒烟测试（每次提交必过）

| # | 用例 | 验证点 |
|:--:|------|------|
| 1 | NT=4096,N=3,L=2,T=8（基线） | 基本功能 + bit-exact |
| 2 | NT=32,N=3,L=2,T=8 | 欠核（token<核数） |
| 3 | NT=256,N=5,L=4,T=16 | 大 W + 多 layer + 长链 |

### 3.3 回归测试（发版前必过）

| # | 用例 | 验证点 |
|:--:|------|------|
| 1 | N=2, T=1（最小 W=1） | 边界最小输出 |
| 2 | NT=65536 | 大 batch，多 tile |
| 3 | 48 核 + 尾核不整除 | 尾核 token 边界 |
| 4 | 非对齐（NT*N 非 8 倍数场景） | DataCopyPad 路径 |

### 3.4 验证方法

```bash
# 环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh
# 数据 + golden
python3 scripts/gen_data.py --nt 4096 --ngram 3 --layers 2 --tables 8
# 构建
cmake -S . -B build && cmake --build build --target engram_hash_custom -j4
# 运行（设备 2）
ASCEND_RT_VISIBLE_DEVICES=2 ./build/engram_hash_custom 4096 3 2 8
# bit-exact 校验
python3 scripts/verify_result.py
# 性能
python3 scripts/benchmark.py
```

---

## 4. 关键里程碑汇总

| 里程碑 | 内容 | 依赖 | 判定 |
|--------|------|------|------|
| M0 | 目录骨架 + CMake 可编译 | Phase 0 | 空 kernel 编译链接通过 |
| M1 | golden + bit-exact 校验脚本 | Phase 1 | 标量重算 == Model.forward |
| M2 | 单核功能跑通 | Phase 2 | 基线输出非空 |
| M3 | 全矩阵 bit-exact | Phase 3 | 多核/多 shape/边界全 pass |
| M4 | PyTorch 集成 | Phase 4 | torch.ops.npu.engram_hash 一致 |
| M5 | 性能达标 | Phase 5 | 满核加速比 ≥0.7×核数，bit-exact 保持 |

---

## 5. 风险与对策

| # | 风险 | 严重度 | 对策 |
|:--:|------|:---:|------|
| 1 | **int64 mul/xor/mod 在 `.asc` 标量单元结果与 PyTorch 不一致**（溢出/符号语义差异） | 高 | 已数值核算：product≤34bit 无溢出、XOR 恒非负、mod 双非负 ⇒ 与 PyTorch 一致；M1 用 tiny case bit-exact 锁定；若发现分歧，用 `AscendC::printf` 打印中间 h 逐步对齐 |
| 2 | **`LocalTensor<int64_t>::GetValue` 在 CANN 9.0.0 行为不确定**（int64 UB 读） | 中 | 兜底：multipliers 走 `__gm__ int64_t*` 直读（量极小，可忽略性能），参考 MTPBlock 小表直读范式；先在 M2 验证 UB int64 读，不稳则切兜底 |
| 3 | **DataCopy 32B 对齐失败**（ngram tile / 输出段非对齐） | 中 | tileTokens 默认对齐 8 倍数（同时满足 N、W 两处基线）；尾 tile 及非对齐 shape 统一走 DataCopyPad；M3 专项验证非对齐用例 |
| 4 | **scalar-bound 导致性能低**（取模标量除法慢、控制开销） | 中 | 满核切分摊薄（P0）；常驻表复用减少 MTE2；循环展开减少控制开销；接受"scalar 是语义必需"，不追求消除 scalar 比例 |
| 5 | **UB 越界**（tileTokens 估算错误 / 输出 outUb 尺寸 L*tileTokens*W 偏大） | 中 | ComputeTiling 严格按 residentUb + perTokenUb 预算并留 MARGIN；M2 用最大 L*T 组合压测 UB 占用；必要时减小 tileTokens |
| 6 | **多核尾核 token 数为 0 或负** | 低 | blockNum 由 ceil 推导，tailTokens 公式含 `bid<blockNum-1` 分支；`myTokens==0` 提前 return |
| 7 | **N=1 退化**（P=0, W=0，空输出） | 低 | kernel 与 host 提前判 W==0 返回空张量（防御性）；测试矩阵不含 N=1 但代码需健壮 |
| 8 | **栈数组按 N 静态分配溢出**（若缓存 mult 行） | 低 | 优先不缓存、直接从 UB/GM 逐元素读；若需缓存按 N≤16 上界静态分配并断言 |
| 9 | **设备可用性**（部分卡 Alarm） | 低 | 固定使用健康卡（NPU 2 / 0 等 OK 状态），`ASCEND_RT_VISIBLE_DEVICES` 显式指定 |

---

## 6. 构建与运行速查

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 生成测试数据 + golden
python3 scripts/gen_data.py --nt 4096 --ngram 3 --layers 2 --tables 8

# 构建（可执行文件 + .so）
mkdir -p build && cd build
cmake -S .. -B .
cmake --build . --target engram_hash_custom -j4     # 直接调用可执行
cmake --build . --target engram_hash_ops -j4        # libengram_hash_ops.so
cd ..

# 运行（设备 2）
ASCEND_RT_VISIBLE_DEVICES=2 ./build/engram_hash_custom 4096 3 2 8
# 参数: numTokens maxNgramSize numLayers numTables [devId] [coreNum]

# bit-exact 校验
python3 scripts/verify_result.py

# 性能
python3 scripts/benchmark.py
```

---

## 7. 参考

| 资源 | 用途 |
|------|------|
| `origin/engram_hash.py` | PyTorch 参考 / golden |
| `test/MTPBlock/` | AIV-only 标量 + `__gm__`/UB 工程模板（kernel/host/torch/CMake） |
| `test/act_quant/docs/{DESIGN,PLAN}.md` | 本仓文档格式范例 |
| DESIGN.md | 架构与 Kernel 详细设计 |
