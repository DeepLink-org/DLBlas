## Overall Design

DLBlas is dedicated to leveraging the latest technologies to achieve the ultimate performance of operators. For example, EP_MoE utilizes cutting-edge industry technologies such as DeepEP and DeepGemm to implement highly efficient MoE modules.

DLBlas is meant to be an operator library for Triton-based operators. As such, kernel developers register their kernels to the library and users ask for a operator by giving operator name and input tensors.

it improves over Triton's autotuner in the following ways:

- **operator selection**: given the same operator, e.g. matmul, there may be different kernel implementations; we want to find the best one based on the input tensors.

- **customized configuration search**: instead of enumerating all possible kernel configurations (BLOCK_SIZE etc.), we want to use advanced algorithm e.g. a bayesian optimizer to search for the best configurations. This needs a flexbile definition of search space and search policy. For DSA hardware, the configuration space is large.

- **caching** the best operator implementation and kernel configurations are cached for the input tensors. It is shape, dtype, device specific.

## Latest News

- 04/24/2026 🚀: The [KernelSwift](https://deeplink.org.cn/kernelswift) fully automatic intelligent operator generation system achieves 100% correctness on NV chips and an average of over 75% correctness on domestic chips, with an average speedup ratio of 3.4x. After manual refinement of the automatically generated operators, an average of 100% correctness is attained on domestic chips, significantly boosting the efficiency of operator development and migration.

## Install

```bash
cd DLBlas
python setup.py install
```

### cute_gemm: CuTe-DSL operator library (vendored)

`dlblas/kernels/cute_gemm/` contains the complete CuTe-DSL operator library
(upstream: [Truth-Ke/DLKernel](https://github.com/Truth-Ke/DLKernel), renamed
to `cute_gemm` in this repository), copied in verbatim as regular files of
this repository — nothing is ever fetched from the upstream repository, and
installs use this local directory only. DLBlas itself is Triton-based;
cute_gemm is a self-contained CuTe-DSL library that keeps its own operators,
autotuner, tune/artifact caches, docs, tests and benchmarks. It is not wired
into DLBlas's OpRegistry or cache.

`python setup.py install` also installs it when an NVIDIA GPU is detected;
other accelerators (for example Ascend) skip it. To install it manually on
such a host (from the repository root):

```bash
pip install -e ./dlblas/kernels/cute_gemm
```

Use it through the DLBlas namespace (pure forwarding, no extra tuning or
copying). The adapter falls back to the in-repo copy when cute_gemm is not
installed:

```python
import dlblas.cute_gemm as cute

out = cute.gemm(a, b)
out = cute.rmsnorm(x, weight, eps=1e-6)
```

cute_gemm's caches are its own (`DLKERNEL_CACHE_DIR`,
`DLKERNEL_ARTIFACT_CACHE_DIR`; see `dlblas/kernels/cute_gemm/docs/`) and are
independent of the DLBlas cache. cute_gemm is licensed under Apache-2.0
(`dlblas/kernels/cute_gemm/LICENSE`); the rest of DLBlas is BSD 3-Clause.

## Getting Started

There are a couple of ways to apply dlblas kernels.

1. get op from dlblas

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

1. import kernel functions from the kernel file

```python
from dlblas.kernels.rms_norm import rms_norm
rms_norm(...)
```

1. import dlblas and use the kernels directly

```python
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
