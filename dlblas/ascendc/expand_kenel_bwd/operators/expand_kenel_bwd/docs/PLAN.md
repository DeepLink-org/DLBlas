# Expand Kernel Backward 算子实现计划 (PLAN.md)

## 1. 需求概述

| 项目 | 内容 |
|------|------|
| 算子名称 | expand_kenel_bwd |
| 需求描述 | 实现 Expand 操作反向传播: 对输入沿 dim=-2 轴做 reduce sum |
| 输入 | `o_grad (n0, n1, mhc_mult, h)` -- FP16 |
| 输出 | `(n0, n1, h)` -- FP16 |
| 典型 shape | `(2, 1024, 4, 1280)` |
| 目标平台 | Ascend 910B2 (DAV_2201), CANN 9.0.0 |
| 算子类型 | Reduction (ARA-FullLoad + Elementwise Add) |

---

## 2. 文件结构规划

```
expand_kenel_bwd/
├── CMakeLists.txt                              # 编译配置 (2 targets)
├── README.md                                   # 算子说明文档
├── run.sh                                      # 一键编译+测试脚本
│
├── op_kernel/
│   ├── expand_kenel_bwd_tiling.h               # Tiling 数据结构 (host/device 共用)
│   └── expand_kenel_bwd_kernel.asc             # Device 侧 Kernel 实现
│
├── op_host/
│   ├── expand_kenel_bwd.asc                    # Host 侧入口 + main (直调验证)
│   └── data_utils.h                            # 文件读写工具函数
│
├── op_extension/
│   ├── ops.h                                   # Torch 接入层函数声明
│   ├── expand_kenel_bwd_torch.cpp              # PyTorch 接入层 (算子逻辑 + Tiling)
│   └── register.cpp                            # TORCH_LIBRARY 注册 + Meta 后端
│
├── scripts/
│   ├── gen_data.py                             # 测试数据生成 (FP16 随机)
│   ├── golden.py                               # Golden 参考计算 (FP32 累加)
│   ├── verify_result.py                        # 直调精度验证
│   └── test_torch.py                           # PyTorch 通路端到端测试
│
├── build/                                      # 构建产物目录
│   ├── expand_kenel_bwd                        # Target 1: 直调可执行文件
│   ├── libexpand_kenel_bwd_ops.so              # Target 2: PyTorch 动态库
│   ├── input/                                  # 测试输入数据 (.bin)
│   ├── output/                                 # 算子输出 + golden (.bin)
│   └── msprof_output/                          # Profiling 原始数据
│
└── docs/
    ├── DESIGN.md                               # 架构设计文档
    ├── PLAN.md                                 # 实现计划文档 (本文件)
    ├── REVIEW.md                               # 代码审查报告
    └── perf/                                   # 性能数据归档
        └── round_001/                          # 第 1 轮 profiling 数据
```

### 编译目标说明

| Target | 类型 | 用途 |
|--------|------|------|
| `expand_kenel_bwd` | 可执行文件 | 直调验证: `./expand_kenel_bwd` 读取 input/*.bin, 写入 output/*.bin |
| `libexpand_kenel_bwd_ops.so` | 动态库 | PyTorch 接入: `torch.ops.load_library(...)` 后通过 `torch.ops.npu.expand_kenel_bwd(x)` 调用 |

---

## 3. 分阶段实现步骤

### 阶段 1: 工程骨架搭建

**目标**: 创建算子项目结构，编译通过空算子。

**任务列表**:
- [x] 创建 `CMakeLists.txt`: 配置两个编译 target, DAV_2201 架构, 依赖库链接
- [x] 创建 `op_kernel/expand_kenel_bwd_tiling.h`: 定义 `ExpandKernelBwdTilingData` 结构体
- [x] 创建 `op_kernel/expand_kenel_bwd_kernel.asc`: 空 kernel 骨架 (Init/Process/CopyIn/Compute/CopyOut 桩函数)
- [x] 创建 `op_host/expand_kenel_bwd.asc`: Host 入口 + main (含 ComputeTiling + KernelCall)
- [x] 创建 `op_host/data_utils.h`: `ReadFile` / `WriteFile` 工具函数
- [x] 编译通过: `cmake .. && make -j4`

**关键配置**:
```cmake
target_compile_options(expand_kenel_bwd PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
```

**验收标准**: `cmake --build .` 成功，两个 target 均编译通过。

---

### 阶段 2: Tiling 实现

**目标**: 实现自适应 Tiling 参数计算，含全载判定和多核分配。

**任务列表**:
- [x] 实现 `ComputeTiling()` 函数:
  - 全载判定: `2×R×tileA0Len×2 + 2×tileA0Len×2 + 2×tileA0Len×4 ≤ 192KB`
  - 计算 `tileA0Len = ((A0 + 127) / 128) * 128`
  - 计算 `a0Outer = ceil(A0 / tileA0Len)`
  - 计算 `totalTiles = A1 × a0Outer`
  - 多核分配: `tilesPerCore`, `tailCoreTiles`, `usedCoreNum`
- [x] 核数动态获取: `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum)`
- [x] 验证 Tiling 参数计算正确性 (手动计算校验)

**关键参数** (以 A1=2048, R=4, A0=1280, 48核 为例):
```
tileA0Len     = 1280  (全载, 1280/128=10 对齐)
a0Outer       = 1
totalTiles    = 2048
tilesPerCore  = ceil(2048/48) = 43
usedCoreNum   = 48
tailCoreTiles = 2048 % 43 = 27
inputSize     = 20,971,520 bytes
outputSize    = 5,242,880 bytes
```

**验收标准**: 各种 shape/核数组合下 Tiling 参数计算正确，无除零、无溢出。

---

### 阶段 3: Device Kernel 实现

**目标**: 实现完整的 device 侧 ARA-FullLoad 归约逻辑。

**任务列表**:
- [x] **TPipe 初始化 + Buffer 分配**:
  ```cpp
  pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, R * tileA0Len * sizeof(half));
  pipe_->InitBuffer(outQueueY, DOUBLE_BUFFER, tileA0Len * sizeof(half));
  pipe_->InitBuffer(castBuf, 2 * tileA0Len * sizeof(float));
  ```
- [x] **多核 tile 范围计算**: `startTile = blockIdx * tilesPerCore`, `endTile` 尾核特殊处理
- [x] **Step 1: CopyIn** -- `DataCopyPad` GM->UB:
  ```cpp
  copyInParams.blockCount = R;
  copyInParams.blockLen   = tileA0Len * sizeof(half);
  copyInParams.srcStride  = (A0 - tileA0Len) * sizeof(half);
  copyInParams.dstStride  = 0;
  ```
- [x] **Step 2: Compute** -- FP32 累加 3 次 Add:
  ```cpp
  // 4 行视图
  LocalTensor<half> row0 = xLocal;
  LocalTensor<half> row1 = xLocal[tileA0Len];
  LocalTensor<half> row2 = xLocal[tileA0Len * 2];
  LocalTensor<half> row3 = xLocal[tileA0Len * 3];
  // FP32 累加
  Cast(accF32, row0, RoundMode::CAST_NONE, count);
  Cast(tmpF32, row1, RoundMode::CAST_NONE, count);
  Add(accF32, accF32, tmpF32, count);
  Cast(tmpF32, row2, RoundMode::CAST_NONE, count);
  Add(accF32, accF32, tmpF32, count);
  Cast(tmpF32, row3, RoundMode::CAST_NONE, count);
  Add(accF32, accF32, tmpF32, count);
  // 截断回 FP16
  Cast(outLocal, accF32, RoundMode::CAST_ROUND, count);
  ```
- [x] **Step 3: CopyOut** -- `DataCopyPad` UB->GM: 单块连续搬出
- [x] **Double Buffer 流水**: `AllocTensor → EnQue → DeQue → FreeTensor` 配对
- [x] **内存管理配对**:
  - `EnQue`/`DeQue`: 2 : 2 (inQueueX + outQueueY)
  - `AllocTensor`/`FreeTensor`: 2 : 2

**验收标准**:
- 编译通过，无警告
- 逻辑审查通过 (REVIEW.md: 99/100)
- 内存管理无泄漏 (EnQue/DeQue, Alloc/Free 配对正确)

---

### 阶段 4: Host 入口与 PyTorch 接入

**目标**: 实现直调验证入口和 PyTorch 算子注册。

**任务列表**:

#### 4.1 直调验证入口 (`op_host/expand_kenel_bwd.asc`)
- [x] ACL 初始化: `aclInit()`, `aclrtSetDevice()`, `aclrtCreateStream()`
- [x] 输入数据读取: `ReadFile()` 从 `input/*.bin` 加载
- [x] Device 内存管理: `aclrtMalloc` / `aclrtMemcpy` / `aclrtFree`
- [x] Tiling 数据搬运: host → device
- [x] Kernel 启动: `expand_kenel_bwd_kernel<<<blockNum, nullptr, stream>>>`
- [x] 输出数据回读 + WriteFile

#### 4.2 PyTorch 接入 (`op_extension/`)
- [x] `expand_kenel_bwd_torch.cpp`: PyTorch Tensor → AscendC kernel 调用适配
  - `TORCH_CHECK` 输入校验 (dtype=FP16, device=NPU, dims=4)
  - 合轴: `A1=n0*n1, R=mhc_mult, A0=h`
  - Tiling 计算 + kernel launch
- [x] `register.cpp`: TORCH_LIBRARY 注册
  - `npu::expand_kenel_bwd(Tensor o_grad) -> Tensor` 算子签名
  - `PrivateUse1` 后端绑定
  - `Meta` 后端注册 (支持 torch.compile / fx)

**验收标准**:
- 直调: `./expand_kenel_bwd` 执行成功，输出 output.bin
- PyTorch: `torch.ops.npu.expand_kenel_bwd(x)` 可正常调用

---

### 阶段 5: 测试策略

#### 5.1 测试数据生成 (`scripts/gen_data.py`)
- [x] 生成 FP16 随机输入: `np.random.randn(n0, n1, mhc_mult, h).astype(np.float16)`
- [x] 计算 golden: FP32 累加后截断回 FP16 (`scripts/golden.py`)
- [x] 输出 `input/input_o_grad.bin` + `output/golden.bin`

#### 5.2 精度测试

| 级别 | 用例 | Shape | 数据类型 | 验证内容 |
|------|------|-------|---------|---------|
| L0 (基础) | 小 shape | (1, 1, 4, 128) | FP16 | 基础功能 + 最小 shape |
| L1 (典型) | 标准 shape | (2, 1024, 4, 1280) | FP16 | 典型生产场景 |
| L1 (典型) | 混合符号 | (2, 512, 4, 256) | FP16 | 正负值混合 |
| L2 (边界) | 全零 | (2, 1024, 4, 1280) | FP16 | 零值稳定性 |
| L2 (边界) | 大值 | (2, 256, 4, 128) | FP16 | 大数值精度 |
| 随机 | 20 seeds | (2, 1024, 4, 1280) | FP16 | 统计稳定性 |

**精度标准** (浮点计算类社区标准):
- FP16 输入, FP32 中间累加
- `rtol = 1e-3`, `atol = 1e-4`
- 验证方式: `np.allclose(result, golden, rtol=1e-3, atol=1e-4)` (转为 FP32 比较)

**实测结果**:

| 测试 | Max Diff | Status |
|------|----------|--------|
| T1 标准 (seed 0-4) | 7.81e-03 | PASS |
| T2 小 shape | 3.91e-03 | PASS |
| T3 零值 | 0.00e+00 | PASS |
| T4 大值 | 0.00e+00 | PASS |
| T5 混合符号 | 6.25e-02 | PASS |
| 随机种子 x20 | 7.81e-03 | 全部 PASS |

#### 5.3 直调验证 (`scripts/verify_result.py`)
- [x] 读取 `output/output.bin` 和 `output/golden.bin`
- [x] 逐元素逐元素精度比对
- [x] 输出 max_diff, 通过/失败判定

#### 5.4 PyTorch 通路测试 (`scripts/test_torch.py`)
- [x] 加载 `libexpand_kenel_bwd_ops.so`
- [x] 调用 `torch.ops.npu.expand_kenel_bwd()`
- [x] 对比 NPU 输出 vs PyTorch `o_grad.float().sum(dim=-2).half()` golden
- [x] 验证 5 个不同 shape 用例

#### 5.5 性能测试 (`msprof`)
- [x] PipeUtilization: MTE2/VEC/MTE3 流水利用率
- [x] ArithmeticUtilization: 向量计算单元利用率
- [x] Memory: 主存带宽 (read/write)
- [x] MemoryL0: L0 缓存行为
- [x] MemoryUB: UB 读写带宽
- [x] L2Cache: L2 缓存命中率
- [x] ResourceConflictRatio: bank conflict / resource conflict

**实测性能摘要**:

| 指标 | 数值 |
|------|------|
| Task Duration | 30.061 us |
| BlockDim | 48 cores |
| 瓶颈类型 | 内存带宽瓶颈 (MTE2 61.3%) |
| Head Overhead | 5.534 us (18.4%) |
| Bank Conflict | 1.90% |

#### 5.6 回归测试
- [ ] 多 shape 组合自动化测试
- [ ] CI 集成 (按需)

---

### 阶段 6: 文档与交付

**任务列表**:
- [x] `README.md`: 算子概述、快速开始、文件结构、API 说明、性能数据、已知限制
- [x] `docs/DESIGN.md`: 架构设计 (需求分析、路线选择、合轴、Tiling、UB 规划、精度、性能模型)
- [x] `docs/PLAN.md`: 实现计划 (本文件)
- [x] `docs/REVIEW.md`: 代码审查报告 (99/100, PASS)
- [x] `docs/perf/round_001/`: 性能数据归档 (7 组 aic-metrics CSV + summary.txt)

---

## 4. 关键 API 选型

### 4.1 Ascend C Kernel API

| API | 来源头文件 | 用途 | 选型理由 |
|-----|----------|------|---------|
| `TPipe` | `kernel_operator.h` | Pipeline 管理 | 标准 Ascend C kernel 入口模式 |
| `TQue<POS, DEPTH>` | `kernel_operator.h` | 双缓冲队列 | Double Buffer 实现 MTE/VEC 流水重叠 |
| `TBuf<>` | `kernel_operator.h` | 临时缓冲区 | FP32 累加中间存储 |
| `DataCopyPad` | `kernel_operator_data_copy_intf.h` | GM↔UB 块式搬移 | 支持 `blockCount`/`srcStride` 参数，适配 ARA 布局 |
| `Cast` | `kernel_operator.h` | 类型转换 | `CAST_NONE` (half→float), `CAST_ROUND` (float→half) |
| `Add` | `kernel_operator_vec_binary_intf.h` | 逐元素向量加法 | FP32 操作, 128 elements/指令 |

### 4.2 Host 侧 API

| API | 用途 |
|-----|------|
| `aclInit` / `aclFinalize` | ACL 初始化和清理 |
| `aclrtSetDevice` / `aclrtResetDevice` | 设备上下文管理 |
| `aclrtCreateStream` / `aclrtDestroyStream` | Stream 管理 |
| `aclrtMalloc` / `aclrtFree` | Device 内存分配 |
| `aclrtMallocHost` / `aclrtFreeHost` | Host 内存分配 |
| `aclrtMemcpy` | Host↔Device 数据搬移 |
| `aclrtSynchronizeStream` | Stream 同步 |
| `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` | 动态获取 Vector 核数 |

### 4.3 PyTorch 接入 API

| API | 用途 |
|-----|------|
| `TORCH_LIBRARY_FRAGMENT(npu, m)` | 算子签名注册 |
| `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` | NPU 后端实现绑定 |
| `TORCH_LIBRARY_IMPL(npu, Meta, m)` | Meta 后端 (torch.compile/fx) |
| `c10_npu::getCurrentNPUStream().stream(true)` | 获取当前 NPU stream (清 queue 模式) |

### 4.4 未选用的 API 及原因

| API | 不选用原因 |
|-----|-----------|
| `ReduceSum` (L2) | 要求归约轴元素连续存储; ARA 布局不满足 |
| `WholeReduceSum` (vcadd) | `srcBlkStride` 语义要求块内连续; ARA 跨行不连续 |
| `BlockReduceSum` | 同上 |
| `REGISTER_TILING_DEFAULT` 宏 | CANN 9.0.0 direct-invoke 不支持 auto-tiling 宏 |
| `GetBlockType()` / `BlockType` | CANN 9.0.0 已移除; 使用 `__vector__` 属性替代 |

---

## 5. 风险点与缓解措施

| 风险 | 等级 | 实际处理 | 状态 |
|------|------|---------|:--:|
| **DataCopyPad srcStride 计算错误** | 中 | 全载时 srcStride=0, 等价于连续搬移; 验证通过 | 已缓解 |
| **Add\<half\> 中间截断导致精度不达标** | 高 | 改为显式 FP32 累加, 精度达标 (max_diff=7.81e-03) | 已解决 |
| **CANN 9.0.0 API 变更 (GetBlockType 移除, DataCopyPadExtParams 签名)** | 中 | `__vector__` 属性替代 AIC 过滤; 四参数 padParams 构造 | 已适配 |
| **实际核数 ≠ 设计假设 24** | 低 | 运行时动态获取, 48 AIV 核自适应 | 已解决 |
| **Tiling auto-tiling 宏不可用** | 中 | Direct-invoke 模式直接指针传递 | 已适配 |
| **R=4 硬编码, 不支持任意 R** | 低 | 当前需求仅 R=4; 扩展改造为循环即可 | 已知限制 |
| **A0 非 128 整数倍** | 低 | Tiling 自动向上对齐 + Padding; 1280 天然对齐 | 已知限制 |
| **仅 FP16, 不支持 FP32/BF16** | 低 | 当前需求 FP16-only; 扩展需 dtype dispatch | 已知限制 |

### 设计偏离记录

| # | 偏离项 | 原设计 | 实际实现 | 原因 |
|---|--------|--------|---------|------|
| 1 | 精度路径 | 直接 `Add<half>` (依赖隐式升精度) | 显式 Cast→FP32→Add→Cast→FP16 | 3 次 Add<half> 中间截断累积误差, FP32 累加单次截断精度更优 |
| 2 | 核类型过滤 | `if (g_coreType == AIC) return;` | `__global__ __vector__` 属性 | CANN 9.0.0 移除 GetBlockType/BlockType API |
| 3 | 核数 | 设计假设 24 AIV 核 | 运行时获取 48 核实际值 | DAV_2201 上 VectorCore 数 = CubeCore × 2 |
| 4 | Tiling 数据传递 | `REGISTER_TILING_DEFAULT` 宏 | 直接 `__gm__` 指针传递 | CANN 9.0.0 direct-invoke 不支持 auto-tiling 宏 |
| 5 | padParams 构造 | `{false, 0, 0, 0}` 三参数 | `{false, 0, 0, static_cast<half>(0.0f)}` 四参数 | CANN 9.0.0 构造函数签名变更 |

---

## 6. 关键里程碑

| 里程碑 | 内容 | 状态 |
|--------|------|:--:|
| M1: 工程骨架 | CMakeLists + tiling.h + 空 kernel + host 入口, 编译通过 | Done |
| M2: Tiling 实现 | ComputeTiling 自适应计算, 参数验证正确 | Done |
| M3: Device Kernel | ARA-FullLoad + FP32 累加 + 3xAdd, 编译通过 | Done |
| M4: 端到端 | 直调验证 + PyTorch 通路, 算子可调用 | Done |
| M5: 精度测试 | 5 用例 + 20 随机种子, 全部通过 (max_diff=7.81e-03) | Done |
| M6: 性能采集 | msprof 7 组 aic-metrics, 30.061us, 内存带宽瓶颈确认 | Done |
| M7: 文档交付 | README + DESIGN + PLAN + REVIEW, 审查 PASS (99/100) | Done |

---

## 7. 依赖项

| 依赖 | 版本 | 用途 |
|------|------|------|
| CANN | 9.0.0 | AscendC 编译工具链 (bisheng) 和运行时 (ACL) |
| CMake | >= 3.16 | 构建系统 |
| Python | >= 3.8 | 测试脚本运行环境 |
| NumPy | >= 1.20 | 测试数据生成和精度验证 |
| PyTorch | >= 2.0 | Golden 参考计算和 PyTorch 通路测试 |
| torch_npu | 匹配 CANN 9.0.0 | NPU 设备支持和算子调用 |

---

## 8. 已知限制

1. **R 硬编码为 4**: 当前使用 3 次 Add 固定处理 R=4 场景。如需支持可变 R (如 mhc_mult=2/8/16)，改为循环: `for (i=0; i<R-1; i++) { Cast(tmpF32, row[i+1], ...); Add(accF32, accF32, tmpF32, ...); }`。注意 `row` 视图数组大小受 UB 限制。
2. **A0 需对齐 128**: `A0_TILE_BASE = 128` 是 `VECTOR_REG_WIDTH / sizeof(half)`。Tiling 会自动向上对齐并 Padding，但非对齐 A0 会引入额外无效计算。
3. **仅支持 FP16**: `sizeof(half)=2` 硬编码在 Tiling 计算和 Buffer 大小中。扩展 FP32/BF16 需修改 Tiling 的 sizeof 计算、Kernel 的 Cast 目标类型、以及 PyTorch 接入层的 dtype 校验。
4. **头开销 18.4%**: 对于 30us 级极短 kernel，5-6us 的 launch/teardown 固定开销占比较高。如需进一步优化，可考虑将多个独立 shape 合并到单次 launch (batch 化)。
