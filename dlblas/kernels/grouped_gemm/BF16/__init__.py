# dlblas/kernels/grouped_gemm/BF16/__init__.py
from .k_grouped_gemm import k_grouped_gemm
from . import utils

import torch
cuda_arch = torch.cuda.get_device_capability()
if cuda_arch[0] >= 9:
    from .m_grouped_gemm_TMA import m_grouped_gemm
else:
    from .m_grouped_gemm import m_grouped_gemm
