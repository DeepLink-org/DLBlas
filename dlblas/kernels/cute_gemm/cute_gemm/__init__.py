__version__ = "0.6.5"

import os

import cute_gemm.dsl as _dlkernel_dsl  # noqa: F401

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from cute_gemm.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

    # Patch before importing any modules that instantiate CuTeDSL. The patch
    # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
    # embedded ptxas-library cubin with one assembled by system ptxas.
    _cute_dsl_ptxas.patch()

# Pythonic CuTe tensor indexing (`:` / `...` sugar) is installed as a side effect
# of importing `cute_gemm.dsl`, which imports `cute_gemm.dsl.cute_tensor_indexing` and
# monkey-patches CuTe's tensor classes process-wide.
from cute_gemm.cross_entropy import (
    cross_entropy,
    cross_entropy_bwd,
    cross_entropy_fwd,
)
from cute_gemm.fused_grouped_gemm import (
    moe_down_proj_dact,
    moe_up_proj_act,
)
from cute_gemm.gemm_base import gemm_with_activation, gemm_with_activation_grad
from cute_gemm.cute_dsl_utils import get_num_sms, set_num_sms
from cute_gemm.gemm_config import TuneMode
from cute_gemm.gemm_interface import gemm, gemm_add
from cute_gemm.gemm_symmetric import gemm_symmetric
from cute_gemm.linear import linear_act_func, linear_func, linear_gated_func
from cute_gemm.linear_cross_entropy import (
    LinearCrossEntropy,
    chunked_linear_cross_entropy,
)
from cute_gemm.mlp import MLP
from cute_gemm.rmsnorm import RMSNorm, rmsnorm
from cute_gemm.rounding import RoundingMode
from cute_gemm.softmax import softmax
from cute_gemm.transform import hadamard_transform

__all__ = [
    "MLP",
    "LinearCrossEntropy",
    "RMSNorm",
    "RoundingMode",
    "TuneMode",
    "chunked_linear_cross_entropy",
    "cross_entropy",
    "cross_entropy_bwd",
    "cross_entropy_fwd",
    "gemm",
    "gemm_add",
    "gemm_symmetric",
    "gemm_with_activation",
    "gemm_with_activation_grad",
    "get_num_sms",
    "hadamard_transform",
    "linear_act_func",
    "linear_func",
    "linear_gated_func",
    "moe_down_proj_dact",
    "moe_up_proj_act",
    "rmsnorm",
    "set_num_sms",
    "softmax",
]
