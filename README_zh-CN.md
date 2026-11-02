## 总体设计

dlBLAS致力于应用最新技术呈现算子的极致性能，例如ep_moe使用DeepEP、DeepGemm等业界最新技术实现高效的moe模块。

dlBLAS 旨在成为一个基于 Triton 的运算符库。因此，内核开发人员可以将其内核注册到该库中，而用户则可以通过提供运算符名称和输入张量来请求运算符。
它通过以下方式改进了 Triton 的自动调谐器:

- **kernel 选择**: 给定相同的运算符，例如 matmul，可能有不同的内核实现；我们希望根据输入张量找到最好的一个。

- **定制配置搜索**: 我们不想枚举所有可能的内核配置（例如 BLOCK_SIZE 等），而是希望使用高级算法（例如贝叶斯优化器）来搜索最佳配置。这需要灵活定义搜索空间和搜索策略。对于 DSA 硬件，配置空间很大。

- **kernel 缓存**：最佳算子实现和内核配置，用于缓存输入张量。其形状、数据类型和设备均特定于特定设备。

## 最新信息

- 04/24/2026 🚀: Day0支持DeepSeekV4算子，[KernelSwift](https://deeplink.org.cn/kernelswift) 全自动生成智能算子生成系统实现NV芯片100%正确性，国产芯片平均实现75+%正确性，平均加速比达3.4x。自动生成的算子经人工修改后国产芯片平均实现100%正确性，大大提高了算子开发和迁移效率。

## 安装

```bash
cd dlBLAS
python setup.py install
```

### cute_gemm：CuTe-DSL 算子库（内置）

`dlblas/kernels/cute_gemm/` 完整内置了 CuTe-DSL 算子库（上游为
[Truth-Ke/DLKernel](https://github.com/Truth-Ke/DLKernel)，在本仓库中
命名为 `cute_gemm`），以本仓库普通文件的形式原样直接拷入——不会从上游
仓库拉取任何内容，安装只使用本地目录。DLBlas 本身基于 Triton；
cute_gemm 是一个自成体系的 CuTe-DSL 算子库，拥有自己的算子实现、
Autotuner、Tune/编译缓存、文档、测试和 Benchmark，不接入 DLBlas 的
OpRegistry 和缓存。

在安装 DLBlas 之外单独安装它（在仓库根目录执行）：

```bash
pip install -e ./dlblas/kernels/cute_gemm
# 或一条命令安装两者：
bash scripts/install_with_cute_gemm.sh
```

通过 DLBlas 命名空间使用（纯转发，不引入额外调优或拷贝）；未安装
cute_gemm 时适配器会自动回退到仓库内副本：

```python
import dlblas.cute_gemm as cute

out = cute.gemm(a, b)
out = cute.rmsnorm(x, weight, eps=1e-6)
```

cute_gemm 的缓存归它自己管理（`DLKERNEL_CACHE_DIR`、
`DLKERNEL_ARTIFACT_CACHE_DIR`，详见 `dlblas/kernels/cute_gemm/docs/`），
与 DLBlas 缓存相互独立。cute_gemm 采用 Apache-2.0 许可
（`dlblas/kernels/cute_gemm/LICENSE`），DLBlas 其余部分为 BSD 3-Clause。

## 开始

有几种方法可以应用 dlblas kernel。

1. 通过get_op导入kernel

```python
from dlblas.utils import get_op
args = parse_args()
dtype = torch.float16
device = 'cuda'
a = torch.randn(
    (args.m, args.k),
    dtype=dtype,
    device=device,
)
b = torch.randn(
    (args.k, args.n),
    dtype=dtype,
    device=device,
)
matmul = get_op('matmul', (a, b))
# test
out = matmul(a, b)
ref_out = a @ b
tol = {
    'atol': 1.0,
}
if torch.allclose(out, ref_out, **tol):
    print('✅ Triton and Torch match')
else:
    print('❌ Triton and Torch differ')
```

1. 从kernel文件导入kernel

```python
from dlblas.kernels.rms_norm import rms_norm
rms_norm(...)
```

1. 导入 dlblas 并直接使用

```python
import dlblas
dlblas.topk_gating(...)
```

## kernel列表

| Kernel              | API                                                                  |
|:-------------------:|:--------------------------------------------------------------------:|
| silu_and_mul        | from dlblas.kernels.activation import silu_and_mul                   |
| add_rms_norm        | from dlblas.kernels.add_rms_norm import call                         |
| rotary_pos_emb      | from dlblas.kernels.apply_rotary_pos_emb import apply_rotary_pos_emb |
| ffn                 | from dlblas.kernels.ffn import call                                  |
| flash_attention_v2  | from dlblas.kernels.flash_attention_v2 import FlashAttentionV2       |
| fp8_gemm            | from dlblas.kernels.fp8_gemm import fp8_gemm                         |
| fused_rotary_and_fa | from dlblas.kernels.fused_rotary_and_fa import FusedRotaryAndFA      |
| partial_rotary_emb  | from dlblas.kernels.partial_rotary_emb import PartialRotaryEmb       |
| topk_gating         | from dlblas.kernels.topk_gating import TopKGatingFunc                |
