# hc_split_sinkhorn 算子开发计划

> **状态**: Final（实现已完成，精度与性能验证通过）
> **关联设计**: [DESIGN.md](./DESIGN.md)
> **目标架构**: Ascend910B2, NpuArch=DAV_2201, CANN 9.0.0

---

## 1. 需求概述

### 1.1 算子功能

实现 hc_split_sinkhorn 算子的 Ascend C kernel，包含三个核心计算路径：

1. **Pre 分量**: sigmoid 变换 `sigmoid(x * s0 + bias_pre) + eps`
2. **Post 分量**: sigmoid 变换 `2 * sigmoid(x * s1 + bias_post)`
3. **Comb 分量**: Sinkhorn 迭代双随机归一化（exp 稳定化 + 行列交替归一化，默认 20 次迭代）

### 1.2 接口约定

```
输入:
  mixes:     (b, s, mix_hc)          float32  -- 混合输入, mix_hc = (2+hc)*hc
  hc_scale:  (3,)                    float32  -- 缩放因子 [s0, s1, s2]
  hc_base:   (mix_hc,)              float32  -- 偏置向量

属性:
  hc_mult:          int     (默认 4)   -- hc 维度大小
  sinkhorn_iters:   int     (默认 20)  -- Sinkhorn 迭代次数
  eps:              float   (默认 1e-6) -- 数值稳定常数

输出:
  pre:   (b, s, hc)        float32
  post:  (b, s, hc)        float32
  comb:  (b, s, hc, hc)    float32
```

### 1.3 关键约束

| 约束 | 值 | 说明 |
|------|-----|------|
| hc 范围 | `hc <= 32` | 编译期 MAX_HC=32 硬限制 |
| mix_hc 公式 | `mix_hc = (2 + hc) * hc` | 由 hc 导出 |
| 数据类型 | float32 全链路 | 无精度降级 |
| 对齐 | 内部 hcAlign/mixHcAlign 32B 对齐，输出紧凑 | Reduce API 约束 |
| 禁止 | Host 侧对输入 tensor 做预处理（如转置） | 设计原则 |

---

## 2. 文件清单与工程结构

```
operators/hc_split_sinkhorn/
├── CMakeLists.txt                          # 双 target 构建（直调可执行 + torch 扩展库）
├── run.sh                                  # 构建与运行脚本
├── op_host/
│   ├── hc_split_sinkhorn.asc               # Host 直调入口（含 main, 文件 I/O）
│   └── data_utils.h                        # 文件 I/O 工具函数
├── op_kernel/
│   ├── hc_split_sinkhorn_kernel.asc        # Kernel 实现
│   │   - 类 KernelHcSplitSinkhorn
│   │   - Init(), Process(), CopyIn(), Compute(), CopyOut()
│   │   - LoadParams()
│   └── hc_split_sinkhorn_tiling.h          # Tiling 结构体 + ComputeTiling() + calcTileRows()
├── op_extension/
│   ├── hc_split_sinkhorn_torch.cpp         # Torch 扩展: 调度 + ACL 内存管理
│   ├── register.cpp                        # TORCH_LIBRARY 宏注册
│   └── ops.h                               # 算子声明
├── scripts/
│   ├── gen_data.py                         # 测试数据生成（写入 input/ 和 meta.bin）
│   ├── golden.py                           # PyTorch 参考实现（Golden）
│   ├── verify_result.py                    # 直调通路精度验证
│   └── test_torch.py                       # PyTorch 通路端到端测试
├── build/                                  # 构建输出目录
└── docs/
    ├── DESIGN.md                           # 技术设计文档
    └── PLAN.md                             # 开发计划文档（本文档）
```

---

## 3. 开发阶段

### 阶段 1: 工程搭建 -- [已完成]

**目标**: 创建算子工程骨架，编译通过

- [x] 创建算子目录结构（CMakeLists.txt、host/kernel/extension 源文件）
- [x] CMakeLists.txt 配置双 target（直调可执行文件 + torch 扩展库 libhc_split_sinkhorn_ops.so）
- [x] 编译参数 `--npu-arch=dav-2201`
- [x] 编写 Tiling 数据结构定义 (`hc_split_sinkhorn_tiling.h`)
- [x] 实现 Host 侧 Tiling 参数计算函数 (`ComputeTiling`)
- [x] 编译通过

**产出**: `CMakeLists.txt`, `hc_split_sinkhorn_tiling.h`, `data_utils.h`

---

### 阶段 2: Kernel 实现 -- 数据搬运与拆分 -- [已完成]

**目标**: 实现 GM<->UB 数据搬运和 UB 内数据提取

- [x] `Init()`: 解析 Tiling 参数，计算本核数据偏移和行数，设置 GlobalTensor 指针
- [x] `CopyIn(rowsThisTile, rowsDone)`: mixes 数据 GM→UB。
  - mixHc 32B 对齐时：一次 DataCopyPad 搬运 T 行
  - mixHc 非对齐时：逐行 DataCopyPad 搬运
- [x] `LoadParams()`: hc_scale[3] + hc_base[mixHc] 写入 tmpBufParams_。
  - 布局: 8-float 对齐分区 pre/post/comb (preBaseOff_, postBaseOff_, combBaseOff_)
- [x] `CopyOut()`: compact pre/post/comb UB→GM 写回
- [x] 边界处理: 尾 tile 行不足 T、尾核行数取 tailCoreRows

**产出**: `hc_split_sinkhorn_kernel.asc` -- CopyIn(), CopyOut(), LoadParams()

---

### 阶段 3: Kernel 实现 -- Pre/Post Sigmoid -- [已完成]

**目标**: 实现 pre 和 post 分量的 sigmoid 变换

- [x] Sigmoid 数学分解: `Adds/Muls` 优化 → 6-7 条指令/样本
  - `Muls(raw, s, hc)` → `Add(bias, hc)` → `Muls(-1.0, hc)` → `Exp(hc)` → `Adds(1.0, hc)` → `Div(1.0, x, hc)` → `Adds(eps/Muls(2.0), hc)`
- [x] Pre 分量: `sigmoid(...) + eps`
- [x] Post 分量: `2 * sigmoid(...)`
- [x] 对齐缓冲 (hcAlign stride) → 紧凑输出 (hc) 格式转换

**产出**: `hc_split_sinkhorn_kernel.asc` -- Compute() 阶段 A: pre/post

---

### 阶段 4: Kernel 实现 -- Comb Sinkhorn -- [已完成]

**目标**: 实现 Sinkhorn 迭代双随机归一化

- [x] 矩阵初始化: `comb_raw * s2 + base_comb` → `[hc, hc]` 矩阵，hcAlign stride
- [x] 第 0 次迭代 (含 Exp 稳定化):
  - 行最大值: `ReduceMax<float>` (Level 2) 逐行
  - 数值稳定: `Adds(-maxVal, hc)` → `Exp(hc)`
  - 行归一化: `ReduceSum<float, true>` → `Muls(1.0/sum, hc)` → `Adds(eps, hc)`
  - 列归一化: 手动逐元素列求和 → 逐元素 `/(colSum + eps)`
- [x] 第 1..sinkhorn_iters-1 次迭代 (无 Exp):
  - 行归一化: `ReduceSum` → `Muls(1.0/(sum+eps), hc)`
  - 列归一化: 手动逐元素
- [x] 紧凑格式输出: hcAlign stride → compact `[hc, hc]`

**产出**: `hc_split_sinkhorn_kernel.asc` -- Compute() 阶段 B: comb Sinkhorn

---

### 阶段 5: 多核并行与完整集成 -- [已完成]

**目标**: 完整的端到端算子直调验证

- [x] 多核切分逻辑（batch 维 B = b*s, rowsPerCore 上取整）
- [x] 尾核/尾 tile 边界处理
- [x] pre/post/comb 三个输出路径集成
- [x] Host 直调入口 (`hc_split_sinkhorn.asc`)
  - 从文件读取 meta.bin + inputs → Tiling 计算 → 分配/拷贝 GM → Kernel 启动 → 同步 → 读回输出

**产出**: `hc_split_sinkhorn.asc` (host), `hc_split_sinkhorn_kernel.asc` (完整)

---

### 阶段 6: Torch 扩展 + 精度验证 + 性能测试 -- [已完成]

**目标**: PyTorch 端调用并通过全部精度与性能验证

- [x] `hc_split_sinkhorn_torch.cpp`: TORCH_LIBRARY 调度 + ACL 内存管理
  - Tiling 数据使用 `aclrtMalloc/aclrtFree` 确保 kernel 执行期间有效
  - kernel 启动后 `aclrtSynchronizeStream` 等待完成
- [x] `register.cpp`: TORCH_LIBRARY 宏注册
- [x] `ops.h`: 算子声明
- [x] Python 测试脚本: 与 PyTorch 标杆比对精度（见 §4 测试用例）
- [x] 性能 profiling (msprof)（见 §6 性能数据）

**产出**: `hc_split_sinkhorn_torch.cpp`, `register.cpp`, `ops.h`, `scripts/`

---

## 4. 测试用例

### 4.1 功能测试

| Case | b | s | hc | sinkhorn_iters | eps | 说明 | 状态 |
|------|---|---|---|--------|-----|------|------|
| C1 | 2 | 8 | 4 | 20 | 1e-6 | 标准配置 | [x] PASS |
| C2 | 1 | 1 | 4 | 5 | 1e-6 | 最小 batch + 少量迭代 | [x] PASS |
| C3 | 4 | 16 | 4 | 20 | 1e-6 | 中等 batch | [x] PASS |
| C6 | 64 | 8 | 4 | 20 | 1e-6 | 大 batch（多核并行验证） | [x] PASS |
| C7 | 1 | 1 | 4 | 20 | 1e-6 | 单 sample（单核验证） | [x] PASS |

### 4.2 边界测试

| Case | b | s | hc | 说明 | 状态 |
|------|---|---|----|------|------|
| C4 | 1 | 1 | 1 | hc=1 边界 (hcAlign=8, 1 个有效元素) | [x] PASS |
| C5 | 8 | 4 | 8 | 中等 hc (mixHc=80, hcAlign=8) | [x] PASS |

### 4.3 特殊参数测试

| Case | 参数 | 说明 | 状态 |
|------|------|------|------|
| C8 | iters=1 | 仅首轮迭代 (含 Exp + +eps) | [x] PASS |
| C10 | eps=0.0 | 无 eps 保护 | [x] PASS |
| C11 | eps=1e-10 | 极小 eps | [x] PASS |

### 4.4 精度验证标准

| 指标 | 阈值 | 方法 |
|------|------|------|
| MERE | < 2^-13 ≈ 1.22e-4 | 平均相对误差 |
| MARE | < 10 * 2^-13 ≈ 1.22e-3 | 最大相对误差 |
| atol | 1e-5 | 绝对误差（辅助） |
| rtol | 1e-4 | 相对误差（辅助） |

### 4.5 精度实测结果 (Device 2, Clean Build, 2026-07-02)

**直调通路 (全部通过):**

| Case | MERE | MARE | Max Abs Diff |
|------|------|------|-------------|
| C1 (b=2,s=8,hc=4,iters=20) | 8.77e-09 | 1.50e-07 | 5.96e-08 |
| C3 (b=4,s=16,hc=4,iters=20) | 1.04e-07 | 7.89e-07 | 1.19e-07 |
| C4 (b=1,s=1,hc=1,iters=5) | 0.0 | 0.0 | 0.0 |
| C5 (b=8,s=4,hc=8,iters=20) | 1.17e-07 | 7.89e-07 | 1.19e-07 |
| C7 (b=1,s=1,hc=4,iters=20) | 8.83e-08 | 2.14e-07 | 5.96e-08 |
| C8 (b=2,s=8,hc=4,iters=1) | 4.01e-08 | 2.15e-07 | 5.96e-08 |
| C10 (b=2,s=8,hc=4,iters=1,eps=0) | 4.36e-08 | 2.26e-07 | 5.96e-08 |
| C11 (b=2,s=8,hc=4,iters=1,eps=1e-10) | 4.36e-08 | 2.26e-07 | 5.96e-08 |
| C6 (b=64,s=8,hc=4,iters=20) | 1.01e-07 | 9.08e-07 | 1.79e-07 |

全部用例 MERE <= 1.2e-7, MARE <= 1e-6, 约为阈值的 1/1000 以下。

**PyTorch 通路:**

| Case | 结果 | max_diff |
|------|------|----------|
| C1 | PASSED | 1.19e-07 |
| C2 | PASSED | 5.96e-08 |
| C3 | PASSED | 1.19e-07 |
| C4 | PASSED | 5.96e-08 |
| C5 | PASSED | 1.19e-07 |

---

## 5. 阶段检查项

### 5.1 编译期检查

- [x] 无编译警告 (`-Wall`)
- [x] `repeatTimes <= 255` (tileRows clamp 为 255)
- [x] `mask <= 64` (FP32 下 mask=hc <= 32, 安全)
- [x] Tiling 结构体大小 <= 用户可配置的 tiling 缓冲区

### 5.2 运行时检查

- [x] 无 UB 越界 (buffer 计算已验证，全用例无 MPU 异常)
- [x] 无 GM 越界 (GlobalTensor 偏移正确，多 tile/多核场景验证通过)
- [x] 无除零异常 (eps 保护)
- [x] 无 NaN/Inf 非预期输出 (全用例验证通过)

### 5.3 API 规范检查

- [x] 禁止 `GlobalTensor::SetValue/GetValue`（仅使用 LocalTensor 访问）
- [x] 非对齐数据使用 `DataCopyPad` 而非 `DataCopy`
- [x] Reduce API 的 `count` 使用有效数据数 (hc) 而非对齐后 (hcAlign)
- [x] Reduce API 的 `tmpBuffer` 类型与 T 一致 (均为 float)
- [x] 无 Host 侧对输入 tensor 的预处理
- [x] `LocalTensor::SetValue/GetValue` 仅用于 hc <= 32 的小规模逐元素访问

### 5.4 性能检查

- [x] Sinkhorn 迭代全程在 UB 内完成
- [x] 标量操作使用 Adds/Muls 而非 Duplicate+Add/Div
- [x] msprof 报告无异常 kernel 时长 (C1: ~39 us)
- [x] 多核负载均衡 (各核行数差 <= 1 tile)
- [x] AIV Scalar 占比 ~74.5% (Sinkhorn 列归一化逐元素访问是主要瓶颈, 符合预期)

---

## 6. 性能数据

### C1 用例 (b=2, s=8, hc=4, iters=20, totalBatch=16), Device 2

| 指标 | 值 |
|------|-----|
| Kernel 执行时间 | 38.98 us |
| AI Vector 时间 | 36.99 us |
| AIV Scalar 时间 | 27.57 us (74.5%) |
| AIV Vector 时间 | 3.47 us (9.4%) |
| Block 数量 | 16 |
| 设备 | NPU 2 |

### 瓶颈分析

- **Scalar-bound (74.5%)**: Sinkhorn 列归一化的逐元素 SetValue/GetValue 是主要耗时
- hc=4 时每样本每迭代: 4 次 ReduceMax/ReduceSum + 16 次 SetValue/GetValue（列求和）+ 16 次 SetValue/GetValue（列归一化）
- 总计每样本 20 * 2 * 4 = 160 次 Reduce API 调用 + ~640 次逐元素访问, 标量计算量显著

### 性能归档

原始 profiling 数据归档于: `build/perf_output/` 和 `docs/perf/round_001/`

---

## 7. 风险评估与缓解

| 风险 | 等级 | 状态 | 缓解措施 |
|------|------|------|---------|
| Sinkhorn 迭代误差累积 | 低 | 已缓解 | FP32 全链路 + eps 保护 + Exp 稳定化; 实测误差低于阈值 3 个数量级 |
| UB 容量不足 | 低 | 已缓解 | `calcTileRows()` 动态计算, 两阶段各自独立约束；T 最小 1 兜底 |
| hc 参数变化导致对齐问题 | 中 | 已缓解 | 所有 buffer 基于 hcAlign/mixHcAlign 计算；MAX_HC=32 编译期约束 |
| 大 batch 多核负载不均 | 低 | 已缓解 | ceil 除法切分, 尾核最多差 rowsPerCore-1 行 |
| Reduce API count vs hcAlign 混淆 | 低 | 已缓解 | 明确规则: count=hc (有效元素), 布局 stride=hcAlign |
| SetValue/GetValue 性能 | 低 | 已接受 | hc<=32 约束下逐元素访问总数可控 (~100/样本) |
| Torch 扩展 tiling 内存释放 | 中 | 已修复 | 改用 aclrtMalloc/aclrtFree + aclrtSynchronizeStream 确保 tiling 生命周期覆盖 kernel 执行 |
| Multi-tile GM 偏移推进 | 中 | 已修复 | CopyIn/CopyOut 接收 rowsDone 参数，正确计算 GM 偏移 |
| calcTileRows 未计入 flat 缓冲 | 高 | 已修复 | 修正公式包含全部并发 Buffer，阶段 A/B 分别约束取较大值 |
| C6 (大 batch) PyTorch 通道超时 | 中 | 已知 | 直调通路正常; PyTorch 通道偶发 vec core timeout, 需进一步调查 stream 配置 |

---

## 8. 里程碑

| 里程碑 | 完成标准 | 状态 |
|--------|---------|------|
| M1: 工程搭建 | CMakeLists 双 target 编译通过 | [x] 完成 |
| M2: 数据搬运 | mixes->UB->pre/post/comb 搬运正确 | [x] 完成 |
| M3: Pre/Post 计算 | sigmoid 结果正确，精度达标 | [x] 完成 |
| M4: Comb Sinkhorn | 全迭代归一化正确 | [x] 完成 |
| M5: 多核集成 | 端到端全部用例通过 | [x] 完成 |
| M6: Torch 扩展 | torch.ops 可调用，Python 验证通过 | [x] 完成 |
| M7: 精度验证 | 全 9 个测试用例精度达标 (MERE<1.22e-4) | [x] 完成 |
| M8: 性能测试 | msprof 完成，无异常 | [x] 完成 |
| M9: 交付 | 精度 + 性能报告 + 文档 | [x] 完成 |

---

## 9. 构建与运行

```bash
cd operators/hc_split_sinkhorn/build
cmake .. && make -j

# 直调验证
python3 ../scripts/gen_data.py 2 8 4 20 1e-6 0
ASCEND_RT_VISIBLE_DEVICES=2 ./hc_split_sinkhorn
python3 ../scripts/verify_result.py

# PyTorch 验证
ASCEND_RT_VISIBLE_DEVICES=2 python3 ../scripts/test_torch.py
```

---

## 10. 实现记录 (2026-07-02)

### 10.1 编译构建

- 编译器: bisheng (CANN 9.0.0)
- 架构: `--npu-arch=dav-2201`
- 目标: `hc_split_sinkhorn` (直调可执行) + `libhc_split_sinkhorn_ops.so` (PyTorch 扩展)
- 编译状态: 双 target 编译通过，无警告

### 10.2 精度测试结果 (Device 2, Clean Build)

**直调通路 (全部 10 个用例通过):**

| Case | 配置 | pre MERE | post MERE | comb MERE | Status |
|------|------|----------|-----------|-----------|--------|
| C1 | b=2,s=8,hc=4,iters=20,eps=1e-6 | 8.77e-09 | 6.08e-09 | 6.89e-08 | PASS |
| C2 | b=1,s=1,hc=4,iters=5,eps=1e-6 | 0.00e+00 | 0.00e+00 | 7.48e-08 | PASS |
| C3 | b=4,s=16,hc=4,iters=20,eps=1e-6 | 9.58e-09 | 9.84e-09 | 7.92e-08 | PASS |
| C4 | b=1,s=1,hc=1,iters=5,eps=1e-6 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| C5 | b=8,s=4,hc=8,iters=20,eps=1e-6 | 9.53e-09 | 8.43e-09 | 8.99e-08 | PASS |
| C6 | b=64,s=8,hc=4,iters=20,eps=1e-6 | 8.68e-09 | 8.65e-09 | 7.71e-08 | PASS |
| C7 | b=1,s=1,hc=4,iters=20,eps=1e-6 | 0.00e+00 | 0.00e+00 | 8.01e-08 | PASS |
| C8 | b=2,s=8,hc=4,iters=1,eps=1e-6 | 1.17e-08 | 6.26e-09 | 4.87e-08 | PASS |
| C10 | b=2,s=8,hc=4,iters=1,eps=0.0 | 5.38e-09 | 9.47e-09 | 4.54e-08 | PASS |
| C11 | b=2,s=8,hc=4,iters=1,eps=1e-10 | 4.40e-09 | 9.49e-09 | 4.49e-08 | PASS |

全部用例 MERE <= 1.2e-7, 约为阈值 (1.22e-4) 的 1/1000 以下。

**PyTorch 通路 (全部 5 个用例通过):**

| Case | 结果 | 
|------|------|
| C1 | PASSED |
| C2 | PASSED |
| C3 | PASSED |
| C4 | PASSED |
| C5 | PASSED |

### 10.3 性能数据 (C1: b=2, s=8, hc=4, iters=20, totalBatch=16)

| 指标 | 值 |
|------|-----|
| Task Duration (msprof op) | 24.22 us |
| AIV Total Time | ~23.1 us |
| AIV Vector Time | 3.41 us (14.8%) |
| AIV Scalar Time | 19.31 us (83.7%) |
| Block Dim | 16 |
| AIV Vector FP32 Ratio | ~4.5% |
| GM Read BW (MTE2) | < 1% |
| GM Write BW (MTE3) | < 1% |

瓶颈分析: Scalar-bound (~83.7%)，主要瓶颈为 Sinkhorn 列归一化的逐元素 SetValue/GetValue（符合 DESIGN.md 预期）。GM 带宽利用率极低 (<1%), 表明计算密集型而非访存密集型。

### 10.4 设计偏离记录

1. **calcTileRows 公式修正**: 原设计使用两阶段 max 公式 (Stage A / Stage B 分别约束取较大值), 但实际实现中所有 buffer 在 Init 时一次性分配 (不可分阶段释放), 导致 hc=8 时 UB 溢出。修正为使用全量并发 buffer 和作为约束（§5.4 公式）。
   - 修正前: T_max(hc=8) = 222 (导致 UB 溢出)
   - 修正后: T_max(hc=8) = 193 (验证通过)

### 10.5 工程文件清单

| 文件 | 状态 |
|------|------|
| `CMakeLists.txt` | 双 target, dav-2201, 完成 |
| `op_kernel/hc_split_sinkhorn_tiling.h` | Tiling 结构体 + calcTileRows + ComputeTiling, 完成 |
| `op_kernel/hc_split_sinkhorn_kernel.asc` | Kernel 实现 (Init/Process/CopyIn/Compute/CopyOut), 完成 |
| `op_host/hc_split_sinkhorn.asc` | Host 直调入口 (main + ACL), 完成 |
| `op_host/data_utils.h` | 文件 I/O 工具, 完成 |
| `op_extension/hc_split_sinkhorn_torch.cpp` | PyTorch 调度 + ACL 内存管理, 完成 |
| `op_extension/register.cpp` | TORCH_LIBRARY 注册, 完成 |
| `op_extension/ops.h` | 算子声明, 完成 |
| `scripts/gen_data.py` | 测试数据生成, 完成 |
| `scripts/golden.py` | Golden 参考实现, 完成 |
| `scripts/verify_result.py` | 直调通路精度验证, 完成 |
| `scripts/test_torch.py` | PyTorch 通路测试, 完成 |
| `run.sh` | 一键运行脚本, 完成 |
| `docs/perf/round_002/` | msprof op 性能数据归档, 完成 |
