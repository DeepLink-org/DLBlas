# apply_mix Ascend C 算子

> **版本**: v5.0 | **日期**: 2026-07-01 | **架构**: per-row Muls+Add + Double Buffer

## 概述

`apply_mix` 算子计算 `(x * mix).sum(-2).bfloat16()`, 即:
1. Broadcast 乘法: `x [n0,n1,mhc,h] (bf16) * mix [n0,n1,mhc,1] (fp32)`
2. Reduction sum: `.sum(-2)` 沿 mhc 维度求和
3. 类型转换: 结果为 bfloat16

## 架构

- **平台**: Ascend910B2 (DAV_2201), CANN 9.0.0
- **技术路线**: SIMD/MemBase, per-row Muls+Add 手动累加 + Double Buffer
- **多核切分**: 沿 A1 维度均分给多核，blockNum <= coreNum
- **计算策略**: fp32 内部计算 + Host/PyTorch 层 bf16<->fp32 转换
- **核心设计**: 
  - 单次 DataCopyPad 多块搬入 x 块 [R, tileA0Len]
  - Per-row Muls<float> 标量广播乘 (就地修改) + Add<float> 向量累加 (R次 Muls + R-1次 Add)
  - TQue<2> Double Buffer 配置 (inQueueX + outQueueY)
  - mix 批量缓存: 仅 a1 变化时重新加载
  - 尾块: Duplicate 零初始化 + PipeBarrier + 逐行 DataCopyPad
- **注意**: DAV_2201 不支持 bf16 SIMD 操作, 采用 fp32 内核

## API 映射

| 功能 | Kernel API | 说明 |
|------|-----------|------|
| 数据搬运 (GM->UB, x块) | `DataCopyPad` (多块) | blockCount=R, blockLen=tLen*4, srcStride=(A0-tLen)*4 |
| 数据搬运 (GM->UB, mix) | `DataCopyPad` (单块) | blockCount=1, blockLen=R*4 |
| 结果搬出 (UB->GM) | `DataCopyPad` (单块) | blockCount=1, blockLen=act*4 |
| 尾块零初始化 | `Duplicate<float>` | xTile 清零, R*alignedCols 元素 |
| V pipe 同步 | `PipeBarrier<PIPE_V>()` | 确保 Duplicate 完成后再 MTE2 操作 (仅尾块) |
| 标量广播乘法 | `Muls<float>` | R 次, 就地修改或初始化累加器 |
| 向量累加 | `Add<float>` | R-1 次, 累加逐行乘法结果 |
| mix 标量提取 | `GetValue(UB)` | 仅 R 次/batch 标量提取, 非逐元素 |
| 流水同步 | `TQue EnQue/DeQue` | inQueueX_ (VECIN, 2) + outQueueY_ (VECOUT, 2) |

## 构建与运行

```bash
# 环境配置
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 完整流程 (编译 + 测试数据生成 + 运行 + 精度验证 + PyTorch 测试)
bash run.sh

# 仅编译
cd build && cmake .. && make -j4

# 仅运行 (复用编译产物)
bash run.sh --skip-build

# 仅 PyTorch 通路
bash run.sh --torch
```

## 精度标准

| 指标 | 阈值 | 数值 |
|------|------|------|
| MERE (平均相对误差) | 2^-7 | 0.00781 |
| MARE (最大相对误差) | 10 x 2^-7 | 0.0781 |

实测定量结果: 全部 7 个测试用例 MERE=0, MARE=0 (位精确)。

## 测试用例

| 用例 | n0 | n1 | mhc | h | 说明 |
|------|----|----|-----|---|------|
| TC1 | 2 | 1024 | 4 | 1280 | 典型 shape |
| TC2 | 1 | 1 | 1 | 64 | 最小 shape |
| TC3 | 1 | 512 | 8 | 256 | 中等 mhc |
| TC4 | 4 | 1 | 4 | 2048 | 大 h, 小 batch |
| TC5 | 1 | 1 | 4 | 1280 | 单 batch |
| TC6 | 2 | 1024 | 4 | 1300 | 非对齐尾块 (h % 64 != 0) |
| TC7 | 1 | 1 | 1 | 1 | 极小值边界 |

## 性能数据 (round_006, msprof op)

| 指标 | 数值 | 目标/预期 |
|------|------|---------|
| Task Duration | 195.52 us | 92-235 us |
| Block Dim | 48 | = coreNum (48) |
| vec_ratio | 4.67% | R=4 极小算量预期 |
| scalar_ratio | 97.22% | SIMD/MemBase per-tile 固有开销 |
| mte2_ratio | 25.60% | 数据搬运 |
| mte3_ratio | 9.00% | 结果搬出 |
| 头开销 | 0.3% | < 10% (优秀) |
| Resource Conflict | 0.99% | < 5% (正常) |
| Current/Rated Freq | 1800/1800 MHz | 满频运行 |

**瓶颈**: Scalar Bound -- per-tile AllocTensor/FreeTensor/EnQue/DeQue 标量开销占主导 (97.22%)。
这是 R=4 极小归约轴的固有特性：每个 tile 仅 7 次向量操作，而 SIMD/MemBase 架构的 per-tile
缓冲区管理开销是固定的。当前设计已使用 Double Buffer + mix caching 最大化流水效率。

## 性能历史

| 版本 | 算法 | 缓冲 | Duration |
|------|------|------|:---:|
| v1.0 | per-row Muls+Add | TQue<1> | 92 us |
| v2.0 | ReduceSum RA | TQue<2> | 242 us |
| v3.0/v4.0 | per-row Muls+Add | TQue<2> | 235 us |
| v5.0 (round_005) | per-row Muls+Add (正式版) | TQue<2> | 233 us |
| v5.0 (round_006) | per-row Muls+Add (正式版) | TQue<2> | 196 us |

## 已知限制

1. **Scalar Bound (97.22%)**: per-tile AllocTensor/FreeTensor/EnQue/DeQue 标量开销是 SIMD/MemBase 架构下 R=4 极小归约轴的固有特性。每个 tile 仅需 7 次向量操作 (4 Muls + 3 Add)，而缓冲区管理开销是恒定的。
2. **Pre-fetch 未实现**: 当前 EnQue->DeQue 紧邻调用限制了 Double Buffer 的重叠效果。重构为预取模式 (prefetch next tile data while computing current tile) 可进一步提升流水效率，但改善幅度受限于极小计算量本身。
3. **vec_ratio 低 (3.92%)**: R=4 时每个 tile 仅 140 次向量指令 (7 次 * 20 repeats)，计算量不足以填满 V pipe。
4. **L2Cache 命中率低 (2.3%)**: 对 x 数据 (A1*R*A0 连续读取) 的 streaming 访问模式导致 L2 miss 率高。启用 L2 CacheMode 或增大搬运粒度可部分缓解。
5. **DAV_2201 bf16 限制**: 不支持 bf16 SIMD 运算。Kernel 内部全 fp32，Host/PyTorch 层完成位转换。
6. **仅 bf16 输出**: 如需 FP16/FP32 输出，需修改 Host/PyTorch 层类型转换逻辑。

## 文件结构

```
operators/apply_mix/
├── op_kernel/
│   ├── apply_mix_tiling.h       # Tiling 结构体和动态参数计算
│   └── apply_mix_kernel.asc     # Kernel 实现 (per-row Muls+Add + Double Buffer)
├── op_host/
│   ├── apply_mix.asc            # ACL 直调 Host 入口 (bf16<->fp32 转换)
│   └── data_utils.h             # 文件读写工具
├── op_extension/
│   ├── apply_mix_torch.cpp      # PyTorch 接入层
│   ├── register.cpp             # TORCH_LIBRARY 注册 (含 Meta backend)
│   └── ops.h                    # 函数声明
├── scripts/
│   ├── gen_data.py              # 测试数据生成
│   ├── golden.py                # Golden 计算 (双通路共用)
│   ├── verify_result.py         # 精度验证 (MERE/MARE)
│   └── test_torch.py            # PyTorch 通路测试
├── CMakeLists.txt               # 构建配置 (双 target)
├── run.sh                       # 一键运行
└── README.md                    # 本文件
```

## PyTorch 使用

```python
import torch
import torch_npu

torch.ops.load_library("build/libapply_mix_ops.so")

x = torch.randn(2, 1024, 4, 1280, dtype=torch.bfloat16).npu()
mix = torch.randn(2, 1024, 4, 1, dtype=torch.float32).npu()
y = torch.ops.npu.apply_mix(x, mix)  # [2, 1024, 1280], bf16
```
