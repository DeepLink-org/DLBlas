"""Reproducer: ptr_shift=False works for TMA load (varlen_k, mainloop A/B).

This test uses the default code path where A/B loads use ptr_shift=False
(2-extra-dim ragged tensor). No pointer shift means globalAddress stays valid.

Expected: PASS
"""
import math
import torch
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

# Default code uses ptr_shift=False for A/B loads (2-extra-dim, no pointer shift)
gemm(A, B, out, None, None, 128, 192, 1, 1,
     pingpong=True, persistent=True, cu_seqlens_k=cu_seqlens_k)
torch.cuda.synchronize()

diff = (out.float() - out_ref.float()).abs().max().item()
status = "PASS" if diff < 0.5 else f"FAIL (max diff={diff:.4f})"
print(f"TMA load with ptr_shift=False (varlen_k A/B): {status}")
