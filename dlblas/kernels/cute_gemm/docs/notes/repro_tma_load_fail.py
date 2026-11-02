"""Reproducer: ptr_shift=True FAILS for TMA load (varlen_k, mainloop A/B).

This test patches create_ragged_tensor_for_tma to force ptr_shift=True for A/B
in a varlen_k GEMM. The TMA load uses the 1-extra-dim ragged tensor with
big_int=2^30, shifting the base pointer ~1GB backward into unmapped GPU memory.
TMA loads validate/access the base pointer, causing a failure.

Expected: FAIL (illegal memory access or wrong results)
"""
import math
import torch

# Monkey-patch to force ptr_shift=True for loads
import cute_gemm.copy_utils as cu
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.cute as cute

original_create = cu.create_ragged_tensor_for_tma.__wrapped__

@dsl_user_op
def create_ragged_ptr_shift_true(T, ragged_dim=0, ptr_shift=True, *, loc=None, ip=None):
    """Force ptr_shift=True regardless of caller's request."""
    return original_create(T, ragged_dim=ragged_dim, ptr_shift=True, loc=loc, ip=ip)

cu.create_ragged_tensor_for_tma = create_ragged_ptr_shift_true

from cute_gemm.gemm import gemm
from cute_gemm.gemm_interface import gemm_ref

torch.random.manual_seed(42)
device = "cuda"

# varlen_k: A is (m, total_k), B is (total_k, n), D is (l, m, n)
m, n, total_k = 2048, 1024, 512
A = torch.randn((m, total_k), device=device, dtype=torch.bfloat16)
A = A.T.contiguous().T  # make m-major
B_orig = torch.randn((total_k, n), device=device, dtype=torch.bfloat16) / math.sqrt(total_k)
B = B_orig.mT  # (n, total_k), n-major
seq_lens = torch.tensor([total_k], device="cpu", dtype=torch.int32)
cu_seqlens_k = torch.cat([
    torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)
]).to(device)

out_ref = gemm_ref(A.float(), B_orig.float(), alpha=1.0, cu_seqlens_k=cu_seqlens_k)
out = torch.empty((1, m, n), dtype=torch.bfloat16, device=device)

try:
    # This forces ptr_shift=True for A/B TMA loads, which should fail
    gemm(A, B, out, None, None, 128, 192, 1, 1,
         pingpong=True, persistent=True, cu_seqlens_k=cu_seqlens_k)
    torch.cuda.synchronize()
    diff = (out.float() - out_ref.float()).abs().max().item()
    status = f"WRONG (max diff={diff:.4f})" if diff >= 0.5 else "PASS (unexpected!)"
except Exception as e:
    status = f"FAIL: {type(e).__name__}: {e}"

print(f"TMA load with ptr_shift=True (varlen_k A/B): {status}")
