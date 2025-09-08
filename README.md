## Overall Design

DLBlas is dedicated to leveraging the latest technologies to achieve the ultimate performance of operators. For example, EP_MoE utilizes cutting-edge industry technologies such as DeepEP and DeepGemm to implement highly efficient MoE modules.

DLBlas is meant to be an operator library for Triton-based operators. As such, kernel developers register their kernels to the library and users ask for a operator by giving operator name and input tensors.

it improves over Triton's autotuner in the following ways:

- **operator selection**: given the same operator, e.g. matmul, there may be different kernel implementations; we want to find the best one based on the input tensors.

- **customized configuration search**: instead of enumerating all possible kernel configurations (BLOCK_SIZE etc.), we want to use advanced algorithm e.g. a bayesian optimizer to search for the best configurations. This needs a flexbile definition of search space and search policy. For DSA hardware, the configuration space is large.

- **caching** the best operator implementation and kernel configurations are cached for the input tensors. It is shape, dtype, device specific.


## Install

```
cd DLBlas
python setup.py install
```
## Getting Started
There are a couple of ways to apply dlblas kernels.
1. get op from dlblas
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
2. import kernel functions from the kernel file
```
from dlblas.kernels.rms_norm import rms_norm
rms_norm(...)

```
3. import dlblas and use the kernels directly
```
import dlblas
dlblas.topk_gating(...)
```
## Low-level APIs
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
