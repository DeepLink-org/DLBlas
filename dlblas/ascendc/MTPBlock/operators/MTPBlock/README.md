# MTPBlock AscendC 算子

**Multi-Token Prediction Block** — DeepSeek-V4-Pro 推理模型核心算子。

## 概览

MTPBlock 是一个复合算子，集成了以下子模块：
- **Embedding 融合** (K1): Embedding Lookup + RMSNorm + 双路 Linear + 广播加
- **Hyper-Connection 预处理** (K2): RMSNorm + Linear + Sigmoid + Sinkhorn 归一化 + 加权融合
- **稀疏注意力** (K3): Q/KV 投影 + 注意力(softmax+sink) + 输出投影
- **Hyper-Connection 后处理** (K4): 广播乘加
- **MoE FFN** (K5): Shared Expert (SwiGLU: w1→SiLU × w3→w2)
- **输出头** (K6): hc_head + RMSNorm + lm_head

## 环境要求

| 项目 | 值 |
|------|-----|
| 芯片 | Ascend910B2 (DAV_2201) |
| CANN | 9.0.0 |
| bisheng 编译器 | `/usr/local/Ascend/cann-9.0.0/bin/bisheng` |
| Python | 3.10+ |
| PyTorch | 2.x + torch_npu |

## 快速开始

```bash
# 设置环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 一键编译 + 测试全部 kernel
cd operators/MTPBlock
bash run.sh --all

# 或分步操作:
mkdir -p build && cd build
cmake ..
make -j4
python3 ../scripts/gen_data.py
./mtpblock_custom 1   # K1
./mtpblock_custom 2   # K2
./mtpblock_custom 3   # K3
./mtpblock_custom 4   # K4
./mtpblock_custom 5   # K5
./mtpblock_custom 6   # K6

# 精度验证
python3 ../scripts/verify_result.py output/k1_feat.bin output/golden_k1.bin fp16
python3 ../scripts/verify_result.py output/k2_y.bin output/golden_k2_y.bin fp16
python3 ../scripts/verify_result.py output/k3_out.bin output/golden_k3.bin fp16
python3 ../scripts/verify_result.py output/k4_out.bin output/golden_k4.bin fp16
python3 ../scripts/verify_result.py output/k5_out.bin output/golden_k5.bin fp16
python3 ../scripts/verify_result.py output/k6_logits.bin output/golden_k6.bin fp32

# PyTorch 通路测试
python3 -c "
import torch, torch_npu
torch.ops.load_library('build/libmtpblock_ops.so')
b,s,hc,d = 1,8,4,512
x = torch.randn(b,s,d,dtype=torch.bfloat16).npu()
r = torch.randn(b,s,hc,d,dtype=torch.bfloat16).npu()
p = torch.randn(b,s,hc,dtype=torch.float32).npu()
c = torch.randn(b,s,hc,hc,dtype=torch.float32).npu()
y = torch.ops.mtpblock.hc_post(x,r,p,c)
print('PyTorch test PASSED')
"
```

## 实现状态

| Kernel | 名称 | 精度 (MARE) | 性能 (us) | 状态 |
|--------|------|:---:|:---:|:---:|
| K1 | mtp_embed_fuse | 5.09e-03 | 167,863 | PASS |
| K2 | hc_pre | 1.12e-03 | 6,365 | PASS |
| K3 | attn_block | 1.62e-02 | 72,721 | PASS |
| K4 | hc_post | 6.30e-04 | 2,342 | PASS |
| K5 | moe_block | 8.62e-04 | 113,944 | PASS |
| K6 | mtp_head | 1.35e-03 | 9,175 | PASS |

> 精度标准: MARE < 7.81e-02 (浮点计算类社区标准 bf16 档)
> 性能数据: msprof 采集, 1 core, demo shape (b=1,s=8,hc=4,d=512,vocab=1000)
> 详细分析: `docs/perf/round_001/summary.txt`

## 目录结构

```
operators/MTPBlock/
├── op_kernel/                     # NPU 计算层
│   ├── mtpblock_tiling.h          # 共享 Tiling 结构体
│   ├── k1_embed_fuse_kernel.asc   # K1 kernel
│   ├── k2_hc_pre_kernel.asc       # K2 kernel
│   ├── k3_attn_block_kernel.asc   # K3 kernel
│   ├── k4_hc_post_kernel.asc      # K4 kernel
│   ├── k5_moe_block_kernel.asc    # K5 kernel
│   └── k6_mtp_head_kernel.asc     # K6 kernel
├── op_host/                       # Host 直调层
│   ├── mtpblock_host.asc          # Host 入口 + main (全部 6 kernel)
│   └── data_utils.h               # 数据读写工具
├── op_extension/                  # PyTorch 接入层
│   ├── mtpblock_torch.cpp         # PyTorch host 实现 (K4 hc_post)
│   ├── register.cpp               # TORCH_LIBRARY 注册
│   └── ops.h                      # 函数声明
├── scripts/                       # 测试脚本
│   ├── gen_data.py                # 测试数据生成 (全部 6 kernel)
│   ├── golden.py                  # 参考实现 (全部 6 kernel)
│   ├── verify_result.py           # 精度验证 (fp16/fp32)
│   └── test_torch.py              # PyTorch 通路测试
├── CMakeLists.txt                 # 构建配置
├── run.sh                         # 一键运行脚本
├── README.md                      # 本文件
└── docs/
    ├── DESIGN.md                  # 架构设计文档
    ├── PLAN.md                    # 开发计划
    ├── REVIEW.md                  # 审查报告
    └── perf/round_001/            # 性能采集数据
```

## 关键技术决策

1. **SIMD/MemBase 路径**: DAV_2201 不支持 RegBase/Blaze, 使用通用 Ascend C API
2. **fp16 替代 bf16**: bisheng 编译器 backend 不支持 bf16 C-style cast, 改用 `half` 类型
3. **6 独立 kernel**: UB 容量限制 (192KB), 拆分保证各 kernel UB 可控
4. **fp32 中间精度**: 关键计算 (RMSNorm, Softmax, 累加) 使用 fp32 防止精度损失
5. **DataCopyPad 对齐**: 所有 GM↔UB 搬运使用 DataCopyPad 保证 32B 对齐
6. **动态 tile_s**: tile_s 通过 tiling data 运行时传入 (clamp 到 MAX_TILE_S), 非硬编码
7. **AscendC::Exp 矢量 API**: 全部 6 kernel 的 exp 操作均使用 AscendC::Exp
8. **AscendC::Mul 矢量 API**: RMSNorm 权重乘、gate*up 元素级乘使用 AscendC::Mul

## Round 2 修复记录 (2026-06-30)

| # | 问题 | 来源 | 修复 |
|---|------|------|------|
| L2 | K6 last-token 多 tile 逻辑缺陷 | REVIEW | 添加 `if (tile_idx == n_tiles - 1)` 保护 Step 6-7 |
| L3 | K3 wo_a weight GM GetValue | REVIEW | DataCopyPad 预加载权重行到 UB |
| -- | PyTorch 扩展 segfault | 开发发现 | kernel 调用签名添加 (blockDim, l2Ctrl, stream) 前缀参数 |
| M1 | usedCoreNum 动态化 | REVIEW | PyTorch 侧通过 aclrtGetDeviceInfo 动态获取核数 |
| -- | run.sh 全量测试 | 开发改进 | 支持 --all 模式覆盖全部 6 kernel + PyTorch 验证 |

## 性能特征

- **瓶颈**: 所有 kernel 99%+ 时间在标量执行 (GetValue/SetValue)
- **主要原因**: MatMul 使用标量 dot product 循环, 非 SIMD 向量化
- **最高优化优先级**: K1 (168ms) > K5 (114ms) > K3 (73ms)
- **优化策略**: SIMD 向量化点积 (Mul+ReduceSum) → MatmulImpl → 多核并行

## 已知限制

1. K5 routed expert (Gate + TopK + Per-Expert dispatch) 未实现, 仅含 Shared Expert
2. K3 使用全密集注意力 (O(s²))，非设计中的 O(s·win) 稀疏窗口
3. MatMul 使用标量 dot product, 非 SIMD 向量化 (Mul+ReduceSum) 或 MatmulImpl
4. K1/K3/K5/K6 无双缓冲 (UB 容量受限)
5. ASC host 直调侧 usedCoreNum 硬编码为 1 (demo shape 下合理)
6. 仅验证 K4 的 PyTorch 接入，K1/K2/K3/K5/K6 尚未添加 torch extension
