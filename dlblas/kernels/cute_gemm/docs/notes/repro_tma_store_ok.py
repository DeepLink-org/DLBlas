"""Reproducer: ptr_shift=True works for TMA store (varlen_m, epilogue D).

This test uses varlen_m where the output tensor D is stored via TMA with the
1-extra-dim ragged tensor approach (ptr_shift=True, big_int=2^30).
The shifted base pointer is ~1GB before the actual allocation, in unmapped memory.
TMA stores do not validate/access the base pointer, so this works.

Expected: PASS
"""
import math
import torch
from cute_gemm.gemm import gemm
from cute_gemm.gemm_interface import gemm_ref

torch.random.manual_seed(42)
device = "cuda"

# varlen_m: A is (total_m, k), B is (l, n, k), D is (total_m, n)
m1, m2, k, n = 512, 1024, 256, 128
total_m = m1 + m2
A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16)
B = torch.randn((2, k, n), device=device, dtype=torch.bfloat16) / math.sqrt(k)
cu_seqlens_m = torch.tensor([0, m1, total_m], device=device, dtype=torch.int32)
out_ref = gemm_ref(A.float(), B.float(), alpha=1.0, cu_seqlens_m=cu_seqlens_m)
out = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)

# D store uses ptr_shift=True internally (1-extra-dim ragged tensor)
gemm(A, B.mT, out, None, None, 128, 128, 1, 1,
     pingpong=False, persistent=True, cu_seqlens_m=cu_seqlens_m)
torch.cuda.synchronize()

diff = (out.float() - out_ref.float()).abs().max().item()
status = "PASS" if diff < 0.5 else f"FAIL (max diff={diff:.4f})"
print(f"TMA store with ptr_shift=True (varlen_m): {status}")
