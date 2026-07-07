# engram_hash AscendC 算子独立审查报告 (REVIEW)

> **审查性质**: 独立第三方审查（从 CMake 配置起全流程重建、重测）
> **审查日期**: 2026-07-01
> **算子**: engram_hash（N-gram embedding 索引哈希，整数计算类）
> **目标芯片**: Ascend 910B2 / DAV_2201 · CANN 9.0.0 · NPU 2 (health OK)
> **技术路线**: AIV-only（Vector 核）+ 标量 int64 计算
> **审查人**: 独立审查 Agent

---

## 0. 总评

| 项目 | 结论 |
|------|------|
| **综合评分** | **93 / 100** |
| 独立编译 | ✅ 通过（clean build，零 warning/error，含 VERBOSE 复核） |
| 精度（bit-exact） | ✅ **全过**：官方 72 例矩阵 + 我方 5 例额外边界 + torch 集成 7 例 + 200 次重复调用，全部 `np.array_equal`（atol=0） |
| 性能 | ✅ 达标：geomean **5.66×** vs torch（复现 summary 5.68×）；核扩展 **47.78×**（1→48 核，99.5% 理想） |
| 规范合规 | ✅ 合规：GM 侧全程 DataCopyPad，无 GlobalTensor GetValue/SetValue 黑名单调用；仅 UB 上 LocalTensor 逐元素访问（允许） |
| 设计一致性 | ✅ 实现严格遵循 DESIGN.md（AIV-only / 标量 int64 / token 切分 / 常驻表 / layer 分段写出） |

**结论**：这是一个**高质量、可交付**的算子实现。精度硬约束（bit-exact）被严格满足并经我方独立复算锁定；多核扩展接近理想线性；代码规范、可读、参数化良好。发现的问题均为 LOW/MEDIUM 级，不影响正确性，主要是性能微优化空间与工程整洁度。

---

## 1. 独立编译结果

从全新 `build_review/` 目录、以 `cmake` 配置起完整重建两个 target：

```
cmake -S . -B build_review -DCMAKE_BUILD_TYPE=Release   → configure done (exit 0)
make engram_hash_custom engram_hash_ops -j$(nproc)      → Built (exit 0)
  - engram_hash_custom       372 KB  (直调可执行)
  - libengram_hash_ops.so    1.5 MB  (TORCH_LIBRARY .so)
```

- `--npu-arch=dav-2201` 编译参数正确注入两个 target 的 ASC 编译。
- VERBOSE 重编译扫描 `warning|error|note`（排除无害的 kineto/NOTFOUND）：**零告警**。
- 仅有的 CMake 提示是 Torch 的 `kineto_LIBRARY-NOTFOUND` 静态库警告，属 torch 安装本身，不影响链接与运行。

**评价**：编译健壮，无隐藏告警，CMake 双目标结构清晰。

---

## 2. 代码质量评分（100 分制）

| 维度 | 权重 | 得分 | 说明 |
|------|:---:|:---:|------|
| **Kernel 实现** | 30 | 28 | 逻辑正确、bit-exact、同步规范；扣分：无 P1/P2 微优化（循环展开/dual-issue），标量吞吐未打满 |
| **Tiling 逻辑** | 20 | 19 | 多核切分 + UB 预算 + 多 tile 覆盖正确，UB 占用全场景 ≤164KB 安全；扣分：4 个 tiling 字段（lastTileTokens/inAligned/outAligned/ngramPos）计算后未被 kernel 使用（dead data） |
| **Host 侧** | 20 | 18 | 直调 main + ComputeTiling 结构清晰，边界处理完备；扣分：torch 路径每次调用 `aclrtMalloc`/`aclrtFree` 微小 tiling（56B），小 shape 下成为延迟地板（~88µs） |
| **PyTorch 集成** | 20 | 20 | TORCH_LIBRARY(npu) + PrivateUse1 + Meta 完整；dtype/device/dim 校验齐全；`.contiguous()` 处理非连续输入；Meta 形状推导正确，torch.compile-ready |
| **文档与脚本** | 10 | 8 | DESIGN/PLAN 详实，脚本覆盖 gen/verify/matrix/torch/bench；扣分：benchmark 核扩展探针方法失真（见 §4） |
| **合计** | 100 | **93** | |

---

## 3. 精度验证结果（bit-exact）

整数计算类，通过标准 = **二进制一致 / 绝对误差 0**（`ops-precision-standard`）。我方独立执行全部验证，**全部通过**：

### 3.1 官方验证矩阵（我方重跑，使用我方 build_review 二进制）
```
[matrix] 72/72 passed   [matrix] ALL BIT-EXACT
```
覆盖：NT∈{32,256,1024,4096,65536} × N∈{2,3,4,5} × L∈{1,2,4} × T∈{1,4,8,16} × cores∈{1,8,24,48}，
含尾核不整除（NT=100/1000）、最小 W=1（N=2,T=1）、最大 W（N=5,T=16）。

### 3.2 我方额外边界（矩阵外，独立设计）
| 用例 | 目的 | 结果 |
|------|------|------|
| NT=512 N=3 **L=3** T=8 | int64 DataCopyPad 奇数 LN=9（72B 非 32B 对齐） | PASS |
| NT=777 N=5 L=1 T=1 | 单层 + 奇数 NT | PASS |
| NT=**131072** N=3 L=2 T=8 | 强制每核多 tile（loops>1） | PASS |
| NT=65536 N=2 L=1 **W=1** | W=1 + 大 NT | PASS |
| NT=333 N=4 L=3 **T=7** | 奇数 T，奇数 W=21 | PASS |

### 3.3 PyTorch 集成（torch.ops.npu.engram_hash）
7 例全 bit-exact，含 W=1 对齐边界（NT=256,N=3,T=1）——即 summary 中记录的"已修复的小 W UB 段 32B 对齐" bug，**复核确认已修复**。

### 3.4 独立算法复算（锁定语义）
我方用纯 numpy int64 独立重写 kernel 三重循环逻辑（含 XOR 前缀链 + 逐 table mod + offset），与 `Model.forward` 对比 5 组 shape（含 tiny N=2）：**全部 `np.array_equal==True`**。同时核实关键不变量：
- `max(ngram)*max(mult)` 恒 < 2³⁴（各 shape 实测 9.0e9~9.3e9），乘积无溢出、bit63=0 → XOR 恒非负；
- vocab_sizes > 0，两操作数非负 → C++ `%` == PyTorch `torch.remainder`。**bit-exact 的数值论证成立**。

---

## 4. 性能分析

### 4.1 端到端加速比（我方复跑 benchmark，NPU 2）
| Shape (NT,N,L,T) | torch (ms) | ascend (ms) | speedup | bit-exact |
|------|:---:|:---:|:---:|:---:|
| 32,3,2,8   | 0.622 | 0.088 | 7.07× | ✓ |
| 256,3,2,8  | 0.616 | 0.088 | 7.03× | ✓ |
| 1024,3,2,8 | 0.612 | 0.089 | 6.90× | ✓ |
| 4096,3,2,8 | 0.658 | 0.096 | 6.85× | ✓ |
| **65536,3,2,8** | 1.554 | 1.195 | **1.30×** | ✓ |
| 4096,5,4,16 | 1.762 | 0.565 | 3.12× | ✓ |
| 256,5,4,16  | 1.850 | 0.096 | 19.38× | ✓ |

**Geomean = 5.66×**（summary 报 5.68×，一致）。全 shape bit-exact。

### 4.2 核扩展（msprof 实测 kernel 时间，NT=65536）
| cores | Task Duration (µs) |
|:---:|:---:|
| 1 | 56950.9 |
| 48 | 1191.9 |

**扩展比 47.78×（99.5% 理想线性）**，精确复现 summary 的 47.8×。48 核负载均衡（各核 1188–1190µs，Block Dim=48）。**M5 里程碑（≥0.7×48=33.6×）达成。**

> ⚠️ 注意：summary 中该 47.8× 来自 **msprof kernel 时间**（正确）。而 `benchmark.py` 内的 `core_scaling` 探针测的是**整可执行 wall-clock**（含 ~1.6s 固定 ACL init），导致 1/8/24/48 核都显示 ~1635ms、**无法体现扩展**。这是脚本方法学缺陷（见 M-2），不影响算子本身。

### 4.3 瓶颈定位（msprof PipeUtilization，48 核）
| 指标 | 值 | 解读 |
|------|:---:|------|
| `aiv_scalar_ratio` | **0.998** | 标量流水几乎 100% 占用 → **scalar-bound** |
| `aiv_vec_ratio` | ~1.5e-5 | Vector 单元几乎闲置（int64 语义使然，符合设计） |
| `aiv_mte2_ratio` | ~8e-4 | 输入搬运可忽略 |
| `aiv_mte3_ratio` | ~1.5e-3 | 输出搬运可忽略 |
| `aiv_scalar_wait_ib` | ~3.16µs | 指令 buffer 等待极小 |
| `aic_time` | NA | 确认 AIV-only，无 Cube |

标量密度分析：**~49 cycle/输出元素**，由 int64 取模 `%`（多周期整数除法）+ 循环控制 + UB 地址计算构成。这是标量单元的物理天花板，符合 DESIGN §8.1 的判断。summary 的 bottleneck 字段（0.998 scalar / 0.0 vec）**准确**。

### 4.4 为何 NT=65536 仅 1.30×
大 token 数下 PyTorch 全向量化 int64 路径效率上升，而标量 kernel 工作量线性增长（每元素固定 ~49 cycle）。这是 scalar-bound 算子的固有特性，DESIGN 已如实记录。**不是缺陷**，但也是当前实现加速比的下界所在。

---

## 5. 规范合规检查

| 检查项 | 结果 | 证据 |
|------|:---:|------|
| GlobalTensor GetValue/SetValue 黑名单 | ✅ 无 | kernel 对 GM 仅用 DataCopyPad；GetValue/SetValue 全部作用于 UB 上的 `LocalTensor`（allowed） |
| DataCopy 32B 对齐 | ✅ 合规 | GM↔UB 全程 `DataCopyPad`（Ext 版），自动处理非对齐；输出 UB 段起址按 `alignUp(tileTokens*W,8)` 补齐到 32B（关键修复点，复核有效） |
| DataCopyPad Ext 版本 | ✅ | 统一 `DataCopyExtParams` + `DataCopyPadExtParams<T>` |
| Host/Kernel 头文件隔离 | ✅ | `engram_hash_compute_tiling.h` 不含 `kernel_operator.h`；torch.cpp 走 host 头 |
| 无 std:: 计算函数 / 无动态分配 | ✅ | kernel 内仅栈标量 + UB Tensor；`align32` lambda 为编译期算术 |
| KERNEL_TASK_TYPE | ✅ | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` |
| 核数不硬编码（kernel 侧） | ✅ | kernel 用 `td->blockNum`/`GetBlockIdx()`；host 侧 `EH_CORE_NUM=48` 仅作平台常量传入 tiling（可移植） |
| EnQue/DeQue 同步 | ✅ | 常驻表载入一次同步；per-tile ngUb/outUb 各自 Alloc→EnQue→DeQue→Free，MTE↔Scalar 屏障正确 |
| int64 UB 读（CANN 9.0.0 风险点） | ✅ 已验证 | `LocalTensor<int64_t>::GetValue` 经 72 例 + 奇数 LN=9 边界验证稳定，设计的 GM 直读兜底未触发即可工作 |

---

## 6. 发现的问题

### HIGH
- **无。** 未发现影响正确性、精度或稳定性的高危问题。

### MEDIUM

**M-1｜torch 路径每次调用 aclrtMalloc/aclrtFree 微小 tiling（延迟地板）**
- 位置：`op_extension/engram_hash_torch.cpp:67-77`
- 现象：每次 `engram_hash()` 调用对 56 字节的 `EngramHashTilingData` 做一次 `aclrtMalloc` + `aclrtMemcpy(H2D)` + `aclrtFree`。设备侧 malloc/free 含隐式同步，成为小 shape 的延迟地板（NT=32 实测 ascend~0.088ms，其中约 85µs 为主机开销，kernel 实算仅数 µs）。
- 影响：小 batch 场景加速比被主机开销稀释；大 batch 可忽略。
- 建议：（a）复用一块常驻 device tiling buffer（首调分配、进程内缓存，按 shape 变更才重刷）；或（b）若 CANN 支持，将 tiling 作为 workspace 由框架管理；或（c）小结构体走 `aclrtMallocHost`+`args` 直传。可显著降低小 shape 延迟。

**M-2｜benchmark.py 核扩展探针方法失真**
- 位置：`scripts/benchmark.py:94-123 (core_scaling)`
- 现象：以整可执行 wall-clock 计时，被 ~1.6s 固定 ACL init 淹没，1/8/24/48 核结果几乎相同（~1635ms），**无法反映真实扩展**。真实扩展需读 msprof kernel 时间（我方实测 47.78×）。
- 影响：误导性能读者；summary 的 `core_scaling` 幸而取自 msprof 而非此探针，故 summary 数值正确。
- 建议：核扩展改用 `msprof op` 提取 `Task Duration(us)`，或在可执行内用 `aclrtEvent` 仅计时 kernel 段（剔除 init/H2D/D2H）。

### LOW

**L-1｜Tiling 结构存在未使用字段（dead data）**
- 位置：`op_kernel/engram_hash_tiling.h` 的 `lastTileTokens` / `inAligned` / `outAligned` / `ngramPos`
- 现象：host 计算并填充，但 kernel 从不读取（kernel 每 tile 由 `myTokens` 现算 `cur`，更健壮）。`inAligned/outAligned` 头注释已标"informational"。
- 影响：无正确性影响，仅结构体略冗余、易让读者误以为 kernel 依赖它们。
- 建议：删除未用字段，或在头文件显式注释"仅诊断打印用，kernel 不依赖"。

**L-2｜kernel 未应用 DESIGN 路线图中的 P2 微优化**
- 位置：`op_kernel/engram_hash_kernel.asc` 内层 T/N 循环
- 现象：无 `#pragma unroll`、无 `__restrict__`。P1（h 提升为循环不变量）实际已隐式做到（h 每 position 算一次，T 循环复用）。
- 影响：标量 dual-issue 率有提升空间（profile 显示 single 124.6µs / dual 261.3µs），但 mod 是硬瓶颈，展开收益有限。
- 建议（可选）：对编译期已知小 T/N 尝试 `#pragma unroll` + 预取 vocab/offset 行到栈数组减少 UB GetValue 次数，实测保留即可；属锦上添花。

**L-3｜直调 main / matrix 未做 ACL 返回码检查**
- 位置：`op_host/engram_hash_host.asc`（`aclrtMalloc`/`aclrtMemcpy` 未校验返回值）
- 现象：`data_utils.h` 提供了 `CALL_RT` 宏但 host main 未使用；失败时会静默继续。
- 影响：仅测试驱动程序，非算子本体；异常设备/OOM 下报错不清晰。
- 建议：测试 main 里对关键 ACL 调用加 `CALL_RT` 或返回码断言，便于定位环境问题。

---

## 7. 改进建议（按优先级）

| 优先级 | 建议 | 预期收益 | 关联问题 |
|:---:|------|:---:|:---:|
| P0 | torch 路径缓存/复用 device tiling buffer，消除每调用 malloc/free | 小 shape 延迟显著下降，小 batch 加速比提升 | M-1 |
| P1 | benchmark 核扩展探针改用 msprof kernel 时间 | 性能报告可信度 | M-2 |
| P2 | 删除 tiling 未用字段或明确标注 | 代码整洁 | L-1 |
| P3 | 内层循环 `#pragma unroll` + vocab/off 行栈缓存（实测取舍） | 标量 dual-issue 小幅提升 | L-2 |
| P3 | 直调 main 加 ACL 返回码检查 | 可诊断性 | L-3 |

> 说明：本算子瓶颈为 scalar-bound 的 int64 取模，已是物理天花板；除 M-1（主机开销）外，kernel 侧提升空间有限。P0 是**唯一有明显收益**的方向。

---

## 8. 审查方法与可复现命令

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
OP=/mnt/data01/zmz/workspace/12agent/waic/build/engram_hash/operators/engram_hash

# 1. 独立编译
cmake -S $OP -B $OP/build_review -DCMAKE_BUILD_TYPE=Release
make -C $OP/build_review engram_hash_custom engram_hash_ops -j$(nproc)

# 2. 全矩阵 bit-exact（72 例）
EH_IO_DIR=$OP python3 $OP/scripts/run_verify_matrix.py    # -> 72/72 ALL BIT-EXACT

# 3. torch 集成
python3 $OP/scripts/test_torch.py                          # -> ALL PASS

# 4. 性能
python3 $OP/scripts/benchmark.py                           # -> geomean 5.66x

# 5. 核扩展 + 瓶颈（真实 kernel 时间）
msprof op --output=/tmp/eh --application="$OP/build_review/engram_hash_custom 65536 3 2 8 0 48"
#   OpBasicInfo.csv Task Duration(us): 48核=1191.9 / 1核=56950.9  -> 47.78x
#   PipeUtilization.csv: aiv_scalar_ratio=0.998, aiv_vec_ratio~0
```

**审查独立性声明**：本报告所有编译、精度、性能数据均由审查方在 build_review/ 全新目录独立重建、重跑取得，未直接采信原工程 build/ 产物或 summary.json 的数值（仅作对照，结论一致）。
