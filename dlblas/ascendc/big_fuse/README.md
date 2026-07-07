# big_fuse — MHC Pre-processing Fused Kernel

MHC (Multi-Head Composition) 预处理融合算子，三核流水线实现四阶段计算。

## 规格

| 项目 | 值 |
|------|-----|
| 算子名称 | big_fuse |
| 架构 | DAV_2201 / Ascend910B2 |
| CANN | 9.0.0 |
| 核函数数量 | **3** (K0: bf16→fp32, K1: MatMul, K2: Vector Post-process) |
| 精度标准 | FP32: MERE < 2^-10, BF16: max abs err < 2^-6 |

## 输入输出

| 张量 | Shape | dtype | 方向 |
|------|-------|-------|------|
| residual | [1, 512, 4, 1280] | bf16 | 输入 |
| fn | [24, 5120] | fp32 | 输入 (权重) |
| mhc_scale | [3] | fp32 | 输入 (权重) |
| mhc_base | [24] | fp32 | 输入 (权重) |
| post_mix | [1, 512, 4, 1] | fp32 | 输出 |
| comb_mix | [1, 512, 4, 4] | fp32 | 输出 |
| layer_input | [1, 512, 1280] | bf16 | 输出 |

## 计算流程

```
K0 (AIV, 48 cores): bf16→fp32 Conversion + Flatten
  residual [1,512,4,1280] bf16 → residual_flat [512,5120] fp32

K1 (AIC, 8 cores): MatMul (Cube)
  residual_flat @ fn^T = raw_mixes [512,24] fp32

K2 (AIV, 43 cores): Vector Post-process
  Phase A: RMS Norm: rsqrt(sqrsum/rgs+eps) * raw_mixes
  Phase B: Split + Sigmoid: scale + bias → sigmoid → pre_mix, post_mix
  Phase C: Sinkhorn: 10-iteration doubly-stochastic normalization → comb_mix
  Phase D: Weighted Apply: sum(residual_bf16 * pre_mix, dim=-2) → layer_input
```

## 快速开始

```bash
# 环境准备
source $ASCEND_HOME_PATH/set_env.sh

# 一键运行（编译 + 测试 + 精度验证）
cd operators/big_fuse
bash run.sh

# 仅运行测试（跳过编译）
bash run.sh --skip-build
```

## 精度 (2026-07-01)

| 输出 | dtype | 指标 | 实测 | 阈值 | 判定 |
|------|-------|------|------|------|------|
| post_mix | fp32 | MERE | 3.90e-4 | 9.77e-4 (2^-10) | **PASS** |
| post_mix | fp32 | MaxAbs | 1.78e-4 | - | - |
| comb_mix | fp32 | MERE | 8.55e-4 | 9.77e-4 (2^-10) | **PASS** |
| comb_mix | fp32 | MaxAbs | 1.89e-4 | - | - |
| layer_input | bf16 | MaxAbs | 7.81e-3 | 1.56e-2 (2 ULP) | **PASS** |

- K0 (bf16→fp32) 转换完全无损 (0 mismatches)
- K1 (MatMul) max abs error = 4.5e-8, 接近 fp32 理论精度
- K2 fp32 输出误差来源: AscendC::Sigmoid/Rsqrt/Exp 硬件实现与 PyTorch CPU 数学库的差异，经 10 次 Sinkhorn 迭代放大
- 绝对误差极小 (~1.8e-4), 对 ML 工作负载无影响
- bf16 输出完全达标

## 性能 (2026-07-01, Round 004)

### 端到端延迟 (wall clock)

| Kernel | Core Type | Cores | 延迟 (us) | 占比 |
|--------|----------|-------|----------|------|
| K0 (bf16→fp32) | AIV | 48 | 440 | 26.4% |
| K1 (MatMul) | AIC | 8 | 111 | 6.7% |
| K2 (Vector) | AIV | 43 | 1112 | 66.9% |
| **Total AscendC** | - | - | **1664** | 100% |

### msprof Task Duration

| Kernel | Core Type | Task Duration (us) | AIC/AIV Time (us) | Task Wait (us) |
|--------|----------|--------------------|--------------------|----------------|
| K0 (bf16→fp32) | AIV | 18.26 | 14.29 (AIV) | 0.0 |
| K1 (MatMul) | AIC | 88.80 | 84.49 (AIC) | 188.49 |
| K2 (Vector) | AIV | 1065.90 | 1052.26 (AIV) | 6103.20 |

### 与 PyTorch 对比

| 实现 | 延迟 (us) | 加速比 |
|------|----------|--------|
| PyTorch NPU | ~2164 | 1.00x |
| AscendC 三核 | ~1664 | **~1.30x** |

### 瓶颈分析

- **K2 (67%)**: scalar sqrsum 归约 (5120 elements/token × GetValue) 和 Sinkhorn scalar row/col norm (10 迭代)
- **K0 (26%)**: bf16→fp32 DataCopyPad + Cast
- **K1 (7%)**: MatMul 已高效 (小矩阵 512×5120→512×24)

## 目录结构

```
operators/big_fuse/
├── CMakeLists.txt            # 构建配置
├── run.sh                    # 一键运行脚本
├── op_kernel/
│   ├── big_fuse_k0.asc       # K0: bf16→fp32 Conversion (AIV)
│   ├── big_fuse_k1.asc       # K1: MatMul (AIC, Cube)
│   └── big_fuse_k2.asc       # K2: Vector Post-process (AIV)
├── op_host/
│   ├── big_fuse.asc          # Host 入口 (三核调度 + Tiling + main)
│   └── data_utils.h          # 文件读写工具
├── op_extension/
│   ├── big_fuse_torch.cpp    # PyTorch 扩展 (待激活)
│   ├── register.cpp
│   └── ops.h
├── tiling/
│   └── big_fuse_tiling.h     # TilingHeader K0/K1/K2 定义
├── scripts/
│   ├── gen_data.py           # 测试数据生成
│   ├── golden.py             # PyTorch 参考实现
│   └── verify_result.py      # 精度验证脚本
└── docs/
    ├── DESIGN.md             # 架构设计文档
    ├── PLAN.md               # 开发计划与结果
    ├── environment.md        # 环境信息
    └── perf/                 # 性能采集数据
```

## 关键修复记录

| 问题ID | 描述 | 修复 |
|--------|------|------|
| **H1** | 硬编码核数 | 使用 PlatformAscendC 动态获取 |
| **H2** | 标量操作替代矢量 API | sigmoid 改用 AscendC::Sigmoid, scale 改用 Muls |
| **H4** | K1 __aicore__ → __cube__ | 修正 kernel type 声明 |
| **H5** | AIV scalar/vector coherency | 所有 scalar↔vector 转换点使用 PipeBarrier<PIPE_ALL> |
| **H5a** | Singleton tile DataCopyPad 溢出 | 确保 tokensPerCore 偶数 (k2CoreNum=43, tpc=12) |

## 已知限制

1. K1 多核利用率低 (8/24 AIC cores): N=24 仅 1 个 N-tile
2. K2 sqrsum 归约使用 scalar 循环 (5120 elements/token), DAV_2201 AIV vcadd 硬件限制为 ~64 fp32/token
3. K2 Sinkhorn 在 4×4 矩阵上使用 scalar row/col norm (M4=4 小于向量 ReduceSum 最小尺寸)
4. 仅支持固定 shape [1, 512, 4, 1280]
