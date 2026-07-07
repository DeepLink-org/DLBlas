# apply_mix 算子开发计划 (PLAN.md)

> **版本**: v5.0 | **日期**: 2026-07-01 | **关联设计**: [DESIGN.md](DESIGN.md)

---

## 一、需求概述

| 项目 | 内容 |
|------|------|
| 算子名称 | `apply_mix` |
| 数学定义 | `output = (x * mix).sum(-2).bfloat16()` |
| 输入 | `x [n0,n1,mhc,h]` (bf16), `mix [n0,n1,mhc,1]` (fp32) |
| 输出 | `output [n0,n1,h]` (bf16) |
| 典型 Shape | `n0=2, n1=1024, mhc=4, h=1280` |
| 目标架构 | Ascend910B2 (DAV_2201), CANN 9.0.0 |
| 算子类型 | 融合算子: Broadcast Mul + Reduction Sum + Type Conversion |
| 合轴模式 | ARA-FullLoad: (A1=2048, R=4, A0=1280) |
| 技术路线 | SIMD/MemBase, per-row Muls+Add 手动累加 |
| 编译参数 | `--npu-arch=dav-2201`, C++17 |

---

## 二、工程结构

```
operators/apply_mix/
├── docs/
│   ├── DESIGN.md                 # 架构设计文档（本 PLAN 的关联文档）
│   ├── PLAN.md                   # 本文件（开发计划）
│   ├── REVIEW.md                 # 审查报告（审查后生成）
│   └── perf/                     # 性能数据归档
├── op_kernel/
│   ├── apply_mix_tiling.h       # Tiling 数据结构 + 动态参数计算 (Host/Kernel 共享)
│   └── apply_mix_kernel.asc     # Device 侧 Kernel 实现
├── op_host/
│   ├── apply_mix.asc            # ACL 直调 Host 入口 (含 bf16<->fp32 转换)
│   └── data_utils.h             # 文件读写工具函数
├── op_extension/
│   ├── apply_mix_torch.cpp      # PyTorch 接入层 (TORCH_LIBRARY 算子实现)
│   ├── register.cpp             # TORCH_LIBRARY 注册
│   └── ops.h                    # 函数声明
├── scripts/
│   ├── gen_data.py              # 测试数据生成
│   ├── golden.py                # Golden 参考计算
│   ├── verify_result.py         # 精度验证 (MERE/MARE)
│   └── test_torch.py            # PyTorch 通路端到端测试
├── CMakeLists.txt               # 构建配置 (两个 target)
├── run.sh                       # 一键编译运行
└── README.md                    # 项目说明
```

### 文件依赖关系

```
apply_mix_tiling.h (TilingData 结构体, ComputeTiling 函数)
        │
        ├──→ apply_mix_kernel.asc   (Device Kernel: 全 fp32 计算)
        │         │
        │         ├──→ apply_mix.asc           (ACL Host: bf16<->fp32 转换, Kernel 启动)
        │         │
        │         └──→ apply_mix_torch.cpp     (PyTorch 接入: torch_npu 调用)
        │                    │
        │                    └──→ register.cpp  (TORCH_LIBRARY 注册)
        │
        └──→ gen_data.py / golden.py / verify_result.py (测试)
```

### 构建目标

| Target | 类型 | 来源文件 | 用途 |
|--------|------|---------|------|
| `apply_mix` | Executable | `op_host/apply_mix.asc` | ACL 直调验证 |
| `libapply_mix_ops.so` | Shared Library | `op_kernel/*.asc + op_extension/*.cpp` | PyTorch 接入 |

---

## 三、实现阶段与检查项

### Phase 1: Tiling 参数设计

**文件**: `op_kernel/apply_mix_tiling.h`

**检查项**:
- [x] 定义常量: `DOUBLE_BUFFER=2`, `UB_SIZE=196608`, `MIN_TILE_A0=64`, `MAX_MHC_R=32`, `UB_OVERHEAD=512`
- [x] 定义 `ApplyMixTilingData` 结构体: `blockNum, A1, R, A0, tileA0Len, alignedCols, totalTiles, tilesPerCore`
- [x] 实现 `ComputeTiling(n0, n1, mhc, h, coreNum)`:
  - [x] R clamp: `if (R > MAX_MHC_R) R = MAX_MHC_R`
  - [x] tileA0Len 动态计算: 基于 UB 容量公式 -> min(maxTile, h) -> 64 对齐
  - [x] alignedCols 32B 对齐计算
  - [x] totalTiles = A1 * ceil(A0 / tileA0Len)
  - [x] tilesPerCore = ceil(totalTiles / coreNum)
  - [x] blockNum 钳制: `blockNum = min(computedBlockNum, coreNum)`
- [ ] 预留: 大 R 场景 RowSplit 分支（条件编译宏）

### Phase 2: Kernel 实现

**文件**: `op_kernel/apply_mix_kernel.asc`

**入口属性**: `extern "C" __global__ __vector__ void apply_mix_kernel(GM_ADDR x, GM_ADDR mix, GM_ADDR y, GM_ADDR tiling)`

**检查项**:

- [x] **Init 阶段**:
  - [x] `TQue<VECIN, 2> inQueueX_` — x 块 Double Buffer
  - [x] `TQue<VECIN, 1> mixQ_` — mix 权重 Single Buffer
  - [x] `TQue<VECOUT, 2> outQueueY_` — 结果 Double Buffer
  - [x] 计算 st_/et_ (起止 tile 索引)
  - [x] GlobalTensor 基地址设置 (xBase_, mBase_, yBase_)

- [x] **Process 主循环** (`for t in [st_, et_)`):
  - [x] 计算 `a1, a0t, a0s, act, isTail`
  - [x] **CopyIn (x)**:
    - [x] 正常块: DataCopyPad 多块搬入 (blockCount=R, blockLen=tLen*4, srcStride=(A0-tLen)*4)
    - [x] 尾块: Duplicate 零初始化 + PipeBarrier<PIPE_V>() + 逐行 DataCopyPad
  - [x] **mix 加载 (caching)**: 仅当 `a1 != prevA1` 时重新加载 mixQ; 用 GetValue 提取到 mixVals[]
  - [x] **Compute**:
    - [x] r=0: `Muls(result, xData, mixVals[0], act)` — 初始化
    - [x] r=1..R-1: `Muls(row, row, mixVals[r], act)` 就地修改 + `Add(result, result, row, act)` 累加
  - [x] **CopyOut**: DataCopyPad 单块搬出 (blockCount=1, blockLen=act*4)
  - [x] **Buffer 回收**: FreeTensor 配对释放

- [x] AllocTensor/FreeTensor 配对: 3:3
- [x] EnQue/DeQue 配对: 3:3
- [ ] 预留: 预取模式 (prefetch next tile) 实现 CopyIn/Compute 重叠

### Phase 3: ACL Host 实现

**文件**: `op_host/apply_mix.asc`

**检查项**:
- [x] 设备/Context 初始化 (`aclrtSetDevice`, `aclrtCreateStream`)
- [x] bf16->fp32 输入转换 (`bf16_to_fp32_cpu`: `(uint32_t)bf16_val << 16`)
- [x] GM 内存分配与 H2D 拷贝
- [x] 动态核数获取: `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)`
- [x] Tiling 参数计算 (`ComputeTiling`) 并拷贝到 Device
- [x] Kernel 启动 (`apply_mix_kernel<<<blockNum, nullptr, stream>>>`)
- [x] `aclrtSynchronizeStream(stream)` 同步
- [x] D2H 结果拷贝与 fp32->bf16 输出转换 (`fp32_to_bf16_cpu`: round-to-nearest-even)
- [x] 资源释放和设备复位

### Phase 4: PyTorch 接入层

**文件**: `op_extension/apply_mix_torch.cpp`, `op_extension/register.cpp`

**检查项**:
- [x] PyTorch 算子实现: `apply_mix_impl(x, mix) -> output`
- [x] bf16->fp32 类型转换（输入）和 fp32->bf16 类型转换（输出）
- [x] 获取当前 Stream、CoreNum 等运行时参数
- [x] Tiling 参数计算与 Device 拷贝
- [x] Kernel 调用、同步与资源释放
- [x] `TORCH_LIBRARY` 算子注册 (`register.cpp`)
- [x] Meta backend 支持 shape 推断
- [ ] 多 Shape 动态输入兼容性测试

### Phase 5: 测试与验证

**测试脚本**:

| 脚本 | 用途 |
|------|------|
| `scripts/gen_data.py` | 生成符合真实数据分布的测试输入 (x: sigmoid -> bf16, mix: softmax -> fp32) |
| `scripts/golden.py` | 计算 fp32 参考输出 |
| `scripts/verify_result.py` | 精度验证 (MERE/MARE, top-10 最差误差) |
| `scripts/test_torch.py` | PyTorch 通路端到端测试 |

**测试用例**:

| Case | n0 | n1 | mhc (R) | h (A0) | 覆盖场景 | 验收 |
|------|----|----|---------|--------|---------|:---:|
| TC-1 | 2 | 1024 | 4 | 1280 | 典型 shape | PASS |
| TC-2 | 1 | 1 | 1 | 64 | 最小 Shape + R=1 边界 | PASS |
| TC-3 | 1 | 512 | 8 | 256 | 中等 mhc | PASS |
| TC-4 | 4 | 1 | 4 | 2048 | 大 h，小 batch | PASS |
| TC-5 | 1 | 1 | 4 | 1280 | 单 batch (A1=1) | PASS |
| TC-6 | 2 | 1024 | 4 | 1300 | 非对齐尾块 | PASS |
| TC-7 | 1 | 1 | 1 | 1 | 极小值边界 | PASS |

**精度验收标准**:

| 指标 | 阈值 | 说明 |
|------|------|------|
| MERE | <= 0.0078125 (2^-7) | 平均相对误差 |
| MARE | <= 0.078125 (10 x 2^-7) | 最大相对误差 |
| NaN/INF | 0 | 输出不得出现非数值 |
| Shape 正确 | [n0, n1, h] | 输出维度正确 |

**编译验收**:
- `cmake .. && make -j4` 通过，零错误零警告
- 双 target 均构建成功: `apply_mix` + `libapply_mix_ops.so`

---

## 四、构建与运行

### 4.1 环境要求

| 项目 | 要求 |
|------|------|
| CANN | 9.0.0 |
| PyTorch | >= 2.1.0 |
| torch_npu | 对应 CANN 版本 |
| NPU 设备 | Ascend910B2 (DAV_2201) |
| 编译器 | bisheng (aarch64) |
| 系统 | Linux aarch64 |

### 4.2 构建命令

```bash
cd operators/apply_mix
rm -rf build && mkdir build && cd build
cmake .. && make -j4
```

### 4.3 运行命令

```bash
# ACL 直调验证
cd operators/apply_mix && bash run.sh

# PyTorch 通路测试
python scripts/test_torch.py
```

---

## 五、风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|:---:|---------|
| R 值超出 UB 容量 | Kernel 栈溢出或 UB OOM | 低 | Host Tiling clamp R <= MAX_MHC_R=32 |
| 尾块非对齐导致数据错误 | 尾块输出错误 | 中 | Duplicate 零初始化 + PipeBarrier + 逐行 DataCopyPad |
| blockNum > coreNum 过度调度 | 性能下降 | 低 | ComputeTiling 中 clamp blockNum <= coreNum |
| mixVals 栈溢出 | 大 R 时栈溢出 | 低 | MAX_MHC_R=32 限制，Tiling 阶段 clamp |
| bf16 精度不足 | MARE 超标 | 低 | 全 fp32 计算路径，仅最后截断 |
| tb_buf 精度异常 (ReduceSum RA) | MARE~10 精度不可用 | - | 已规避: 使用 per-row Muls+Add |
| PopStackBuffer 标量瓶颈 | 性能退化 2-3x | - | 已规避: 使用 per-row Muls+Add |

---

## 六、附录：关键 API 调用清单

| 序号 | API 调用 | 所在阶段 | 说明 |
|------|---------|---------|------|
| 1 | `pipe_->InitBuffer(inQueueX_, 2, R * alignedCols * 4)` | Init | x 块 Double Buffer |
| 2 | `pipe_->InitBuffer(mixQ_, 1, R * 4)` | Init | mix Single Buffer |
| 3 | `pipe_->InitBuffer(outQueueY_, 2, alignedCols * 4)` | Init | 结果 Double Buffer |
| 4 | `Duplicate<float>(xTile, 0.0f, R * alignedCols)` | Process (尾块) | 尾块零初始化 |
| 5 | `PipeBarrier<PIPE_V>()` | Process (尾块) | V pipe 同步后 MTE2 |
| 6 | `DataCopyPad(xTile, xRowGm, {R, tLen*4, stride, 0}, {false,0,0,0})` | Process (正常块) | x 多块搬入 |
| 7 | `DataCopyPad(row, rowGm, {1, act*4, 0, 0}, {false,0,0,0})` | Process (尾块逐行) | 尾块逐行搬入 |
| 8 | `DataCopyPad(mBuf, mixGm, {1, R*4, 0, 0}, {false,0,0,0})` | Process (mix) | mix 权重搬入 |
| 9 | `mData.GetValue(r)` | Process (mix) | mix 标量提取 |
| 10 | `Muls<float>(result, xData, mixVals[0], act)` | Process (Compute) | r=0 初始化乘法 |
| 11 | `Muls<float>(row, row, mixVals[r], act)` | Process (Compute) | 就地标量广播乘 |
| 12 | `Add<float>(result, result, row, act)` | Process (Compute) | 向量累加 |
| 13 | `DataCopyPad(yGm, yData, {1, act*4, 0, 0})` | Process (CopyOut) | 结果搬出 |
| 14 | `inQueueX_.EnQue(xTile)` / `inQueueX_.DeQue<float>()` | Process | Double Buffer 流水同步 |
| 15 | `outQueueY_.EnQue(result)` / `outQueueY_.DeQue<float>()` | Process | Double Buffer 流水同步 |
| 16 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` | Host | 动态获取核数 |
| 17 | `ComputeTiling(n0, n1, mhc, h, coreNum)` | Host | Tiling 参数计算 |
| 18 | `apply_mix_kernel<<<blockNum, nullptr, stream>>>` | Host | Kernel 启动 |
| 19 | `bf16_to_fp32_cpu` / `fp32_to_bf16_cpu` | Host | 类型转换 |

---

## 七、版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v1.0 | 2026-06 | 初版: per-row Muls+Add, TQue<1>, 硬编码 tileA0Len |
| v2.0 | 2026-06 | ARA-FullLoad + ReduceSum RA + TQue<2> + 动态 tiling |
| v3.0 | 2026-07 | per-row Muls+Add + TQue<2> + blockNum 钳制 |
| v4.0 | 2026-07 | 文档统一，清理 v2.0 残余 |
| **v5.0** | **2026-07** | **本版: 重新规划，整合历史经验为正式开发计划** |

---

## 八、实现结果 (v5.0 最终验证 -- 2026-07-01)

### 8.1 构建结果

| Target | 状态 | 产物 |
|--------|:---:|------|
| `apply_mix` (Executable) | PASS | `build/apply_mix` |
| `libapply_mix_ops.so` (Shared Library) | PASS | `build/libapply_mix_ops.so` |
| 编译参数 | -- | `--npu-arch=dav-2201`, C++17, 零警告 |

### 8.2 功能测试结果

全部 7 个测试用例通过，位精确 (MERE=0, MARE=0):

| Case | n0 | n1 | mhc (R) | h (A0) | ACL 直调 | PyTorch 通路 |
|------|----|----|---------|--------|:---:|:---:|
| TC-1 | 2 | 1024 | 4 | 1280 | PASS (MERE=0) | PASS (MERE=0) |
| TC-2 | 1 | 1 | 1 | 64 | - | PASS (MERE=0) |
| TC-3 | 1 | 512 | 8 | 256 | - | PASS (MERE=0) |
| TC-4 | 4 | 1 | 4 | 2048 | - | PASS (MERE=0) |
| TC-5 | 1 | 1 | 4 | 1280 | - | PASS (MERE=0) |
| TC-6 | 2 | 1024 | 4 | 1300 | - | PASS (MERE=0) |
| TC-7 | 1 | 1 | 1 | 1 | - | PASS (MERE=0) |

验收结论: 全部精度达标（MERE <= 0.00781, MARE <= 0.0781），NaN/INF=0。

### 8.3 性能采集结果 (round_006, msprof op)

| 指标 | 数值 | 评估 |
|------|------|------|
| Task Duration | 195.52 us | 符合预期范围 (92-235 us) |
| Block Dim | 48 | 满核利用 |
| Current/Rated Freq | 1800/1800 MHz | 无 DVFS 降频 |
| aiv_vec_ratio | 4.67% (avg) | 极低，符合 R=4 极小计算量预期 |
| aiv_scalar_ratio | 97.22% (avg) | 极高，SIMD/MemBase per-tile 标量开销固有 |
| aiv_mte2_ratio | 25.60% (avg) | 搬运占比适中 |
| aiv_mte3_ratio | 9.00% (avg) | 输出搬运占比低 |
| 头开销 | 0.68 us (0.3%) | 优秀，几乎无启动开销 |
| Resource Conflict | 0.99% total | 低，无 bank conflict 问题 |
| L2Cache hit | 2.3% | 低，但对该小数据量场景影响有限 |
| vec_wait | 24.68% | 部分流水重叠存在 |

**SCALAR 子类耗时 (avg)**: single=4.51us, dual=3.88us, wait=5.25us, wait_ib=0.13us.

**性能瓶颈判定**: Scalar Bound -- per-tile AllocTensor/EnQue/DeQue/FreeTensor 标量开销占主导 (97.22%)。这是 R=4 极小归约轴的固有特性：每个 tile 仅 7 次向量操作，而 SIMD/MemBase 架构的 per-tile 缓冲区管理开销是固定的。

**优化空间**: 有限。进一步优化方向包括：
- **预取模式** (prefetch next tile): 在计算当前 tile 时预取下一个 tile 的数据，使 CopyIn/Compute/CopyOut 真正重叠。但当前架构下 EnQue->DeQue 紧邻调用限制了重叠效果。
- **增大 tileA0Len**: 当前已最大化 (tileA0Len=1280 = 完整 A0 维度)，无提升空间。

**性能历史**:

| 轮次 | Task Duration | vec_ratio | scalar_ratio | 备注 |
|------|:---:|:---:|:---:|------|
| round_005 | 232.94 us | 3.92% | 97.64% | 初版采集 |
| round_006 | 195.52 us | 4.67% | 97.22% | v5.0 正式验证 |

详细性能数据归档: `docs/perf/round_006/`

### 8.4 DESIGN.md 合规性自检

| # | 约束/设计决策 | 状态 | 验证方式 |
|---|-------------|:---:|---------|
| C1 | SIMD/MemBase 架构 (TPipe+TQue+DataCopyPad) | PASS | 代码审查: apply_mix_kernel.asc |
| C2 | Kernel 全 fp32, Host/PyTorch 层 bf16<->fp32 | PASS | 代码审查: apply_mix.asc, apply_mix_torch.cpp |
| C3 | 禁止 Host 侧结构预处理 | PASS | 仅元素级位转换 (bf16_to_fp32_cpu) |
| C4 | 32B 对齐 (alignedCols) | PASS | ComputeTiling 中 alignedCols 计算 |
| C5 | blockNum <= coreNum | PASS | ComputeTiling blockNum 钳制逻辑 |
| C6 | Double Buffer (TQue<2>) | PASS | inQueueX_ 和 outQueueY_ 均为 TQue<..., 2> |
| C7 | R <= MAX_MHC_R (32) | PASS | ComputeTiling R clamp 逻辑 |
| C8 | repeatTimes <= 255 | PASS | tileA0Len=4864 → repeatTimes=76 ≤ 255 |
| C9 | 禁止结构变换 | PASS | 无 transpose/reshape 操作 |
| -- | per-row Muls+Add (非 ReduceSum RA) | PASS | Kernel 使用手动 Muls+Add 累加 |
| -- | mix caching (batch 变化时重载) | PASS | Process() 中 prevA1 比较逻辑 |
| -- | 尾块 Duplicate+PipeBarrier+逐行 DataCopyPad | PASS | isTail 分支逻辑 |
| -- | 多核沿 A1/A0 均分 | PASS | ComputeTiling totalTiles/tilesPerCore |
| -- | 动态 tileA0Len 计算 | PASS | UB 容量公式在 ComputeTiling 中 |
| -- | TQue mixQ<1> (Single Buffer) | PASS | mixQ_ 配置为 TQue<VECIN, 1> |

**结论**: 所有 DESIGN.md 规定的设计框架和关键决策已在代码中完整实现，无偏离。
