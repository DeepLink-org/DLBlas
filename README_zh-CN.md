## 总体设计
dlBLAS致力于应用最新技术呈现算子的极致性能，例如ep_moe使用DeepEP、DeepGemm等业界最新技术实现高效的moe模块。

dlBLAS 旨在成为一个基于 Triton 的运算符库。因此，内核开发人员可以将其内核注册到该库中，而用户则可以通过提供运算符名称和输入张量来请求运算符。
它通过以下方式改进了 Triton 的自动调谐器:

- **kernel 选择**: 给定相同的运算符，例如 matmul，可能有不同的内核实现；我们希望根据输入张量找到最好的一个。

- **定制配置搜索**: 我们不想枚举所有可能的内核配置（例如 BLOCK_SIZE 等），而是希望使用高级算法（例如贝叶斯优化器）来搜索最佳配置。这需要灵活定义搜索空间和搜索策略。对于 DSA 硬件，配置空间很大。

- **kernel 缓存**：最佳算子实现和内核配置，用于缓存输入张量。其形状、数据类型和设备均特定于特定设备。


## 安装

```
cd dlBLAS
python setup.py install
```
## 开始
有几种方法可以应用 dlblas kernel。
1. 通过get_op导入kernel
```
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
2. 从kernel文件导入kernel
```
from dlblas.kernels.rms_norm import rms_norm
rms_norm(...)

```
3. 导入 dlblas 并直接使用
```
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
