# mhc_post AscendC 算子

## 概述

Multi-Head Compression Post-processing 算子，实现批量小矩阵乘 (K=4) + 广播乘加 + bf16 输出。

**数学公式**:
```
term2[a,b,m,:] = Σ_k comb_res_mix[a,b,m,k] × residual[a,b,k,:]
output[a,b,m,:] = x[a,b,:] × post_layer_mix[a,b,m,0] + term2[a,b,m,:]
输出 cast 为 bf16
```

对应 PyTorch 参考实现 (`origin/mhc_post.py`):
```python
term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
output = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
```

## 规格

| 项目 | 值 |
|------|-----|
| 算子名称 | mhc_post |
| x | (n0, n1, h) bfloat16 |
| residual | (n0, n1, M=4, h) bfloat16 |
| post_layer_mix | (n0, n1, M=4, 1) float32 |
| comb_res_mix | (n0, n1, M=4, M=4) float32 |
| output | (n0, n1, M=4, h) bfloat16 |
| 默认 shape | n0=2, n1=4096, M=4, h=1280 |
| 目标平台 | Ascend910B2 (DAV_2201), CANN 9.0.0 |
| 技术路线 | SIMD/MemBase (TPipe + TQue + Vector API) |
| 多核切分 | 沿 n1 维度均匀切分，最多 20 核 |

## 构建与运行

```bash
# 准备环境
source ${ASCEND_HOME_PATH}/set_env.sh

# 一键编译 + 标准测试 (TC-01)
bash run.sh

# 运行全量测试 (TC-01 ~ TC-12)
bash run.sh --test-all

# 运行单个用例
bash run.sh --test=TC-06

# 跳过编译，直接运行
bash run.sh --skip-build

# PyTorch 接入测试
python3 scripts/test_torch.py
```

## 测试结果

### 功能测试 (12/12 全部通过)

| TC | Shape | 说明 | 状态 |
|----|-------|------|------|
| TC-01 | (2,4096,1280,4) | 标准配置 | PASS |
| TC-02 | (1,1,64,4) | 极小 shape | PASS |
| TC-03 | (1,1,1,4) | 最小 h | PASS |
| TC-04 | (1,16,1280,4) | 小 n1 | PASS |
| TC-05 | (2,4096,64,4) | h=C_TILE | PASS |
| TC-06 | (2,4097,1280,4) | n1 不整除 | PASS |
| TC-07 | (1,1,1280,4) | n1<核数 | PASS |
| TC-08 | (2,4096,1280,4) | 全零输入 | PASS |
| TC-09 | (2,4096,1280,4) | 极大值 | PASS |
| TC-10 | (2,4096,1280,4) | 混合正负值 | PASS |
| TC-11 | (1,4096,1280,4) | n0=1 | PASS |
| TC-12 | (2,4096,130,4) | h 非整除 | PASS |

### 精度标准

```
MERE = avg(|output - golden| / (|golden| + 1e-7))
MARE = max(|output - golden| / (|golden| + 1e-7))   [|golden| > 1e-5]

bf16 输出: MERE < 2^-7 (~0.00781) AND MARE < 10 * 2^-7 (~0.0781)
```

标准 shape (TC-01): MERE = 1.4e-7, MARE = 7.8e-3 → **PASS**

### PyTorch 接入测试 (4/4 通过)

```bash
python3 scripts/test_torch.py
```

## 性能数据 (Round 002, 双缓冲 v2)

| 指标 | 值 |
|------|-----|
| 总延迟 | 8,691 us |
| AIV vec | 46.70% |
| AIV scalar | 99.00% (瓶颈：系数 GetValue 调用) |
| AIV MTE2 | 28.60% |
| AIV MTE3 | 18.80% |

**瓶颈**: SCALAR BOUND - 每 tile 需 20 次 `GetValue()` 读取系数标量，累积 ~164,000 次/核。

## 文件结构

```
operators/mhc_post/
├── CMakeLists.txt                   # 双 Target: 可执行文件 + PyTorch .so
├── run.sh                           # 编译+运行+测试 (--test-all, --test=TC-XX)
├── op_kernel/
│   ├── mhc_post_tiling.h            # Tiling 常量与结构体 (kernel+host 共用)
│   ├── mhc_post_kernel.asc          # Kernel 实现 (双缓冲 TQue + TBuf + Vector API)
│   └── mhc_post_kernel_decl.h       # Kernel 函数声明
├── op_host/
│   ├── mhc_post.asc                 # Host 入口 + main (直调验证)
│   └── data_utils.h                 # 文件读写工具
├── op_extension/
│   ├── mhc_post_torch.cpp           # PyTorch 接入层 (stream(true) 模式)
│   ├── register.cpp                 # TORCH_LIBRARY 注册 (npu::mhc_post)
│   └── ops.h                        # 函数声明
├── scripts/
│   ├── gen_data.py                  # 测试数据生成 (支持 --test TC-XX, --all)
│   ├── golden.py                    # 参考输出计算 (fp32 sequential 匹配 kernel)
│   ├── verify_result.py             # 精度验证 (MERE/MARE)
│   └── test_torch.py                # PyTorch 接入端到端测试
└── docs/
    ├── DESIGN.md                    # 架构设计文档
    ├── PLAN.md                      # 开发计划与测试结果
    └── perf/round_002/              # v2 性能数据 (msprof)
```

## PyTorch 接入

```python
import torch
import torch_npu
import sys
sys.path.insert(0, 'operators/mhc_post')

torch.ops.load_library("operators/mhc_post/build/libmhc_post_ops.so")

# 调用算子
x    = torch.randn(2, 4096, 1280, dtype=torch.bfloat16).npu()
res  = torch.randn(2, 4096, 4, 1280, dtype=torch.bfloat16).npu()
pm   = torch.randn(2, 4096, 4, 1, dtype=torch.float32).npu()
cmb  = torch.randn(2, 4096, 4, 4, dtype=torch.float32).npu()

output = torch.ops.npu.mhc_post(x, res, pm, cmb)
# output: (2, 4096, 4, 1280) bfloat16
```

## 关键设计要点

1. **Vector 路线**: K=4 极短向量点积，选 Vector API (Muls + Add) 优于 Cube 16×16 MAC (利用率仅 25%)
2. **双缓冲 TQue**: `TQue<VECIN/VECOUT, 2>` 实现三阶段 CopyIn/Compute/CopyOut 流水重叠
3. **TBuf 系数**: `comb_res_mix` (64B) + `post_layer_mix` (16B) 逐 batch 加载到 TBuf，全列 tile 共享
4. **bf16 精度路径**: 输入 bf16 → Cast NONE → fp32 计算 → Cast ROUND → 输出 bf16
5. **动态核数**: `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 运行时获取，最大 20 核
6. **n1 维度切分**: 均匀切分，尾核自动适应，空闲核直接跳过
