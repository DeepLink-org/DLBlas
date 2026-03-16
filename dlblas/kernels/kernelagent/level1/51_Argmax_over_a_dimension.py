import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _argmax_row_kernel(
    X_ptr,          # *T [B, K] contiguous
    OUT_ptr,        # *int64 [B]
    B,              # rows
    K,              # reduction length
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= B:
        return

    # Row base pointer offset
    row_base = (pid * K).to(tl.int64)

    neg_inf = -float("inf")
    best_val = tl.full((), neg_inf, dtype=tl.float32)   # scalar
    best_idx = tl.zeros((), dtype=tl.int64)             # scalar

    k0 = 0
    while k0 < K:
        idx = tl.arange(0, BLOCK_K)
        k_idx = k0 + idx
        mask = k_idx < K

        offs = row_base + k_idx.to(tl.int64)
        vals = tl.load(X_ptr + offs, mask=mask, other=neg_inf, cache_modifier=".cg").to(tl.float32)

        # Max value in this tile and the first index where it's reached
        tile_max = tl.max(vals, axis=0)
        eq = (vals == tile_max) & mask

        big = (K + 1)
        big_vec = tl.full((BLOCK_K,), big, dtype=tl.int64)
        cand_idx = tl.where(eq, k_idx.to(tl.int64), big_vec)
        tile_first_idx = tl.min(cand_idx, axis=0)

        # Update global best: prefer strictly greater; on ties, prefer smaller index
        greater = tile_max > best_val
        equal = tile_max == best_val
        earlier = tile_first_idx < best_idx
        update = greater | (equal & earlier)

        best_val = tl.where(update, tile_max, best_val)
        best_idx = tl.where(update, tile_first_idx, best_idx)

        k0 += BLOCK_K

    tl.store(OUT_ptr + pid, best_idx)


class ModelNew(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        # Fallback to PyTorch for CPU tensors or edge cases; preserves exact semantics
        if not x.is_cuda:
            return torch.argmax(x, dim=self.dim)

        if x.ndim == 0:
            return torch.argmax(x, dim=self.dim)

        d = self.dim if self.dim >= 0 else self.dim + x.ndim
        if d < 0 or d >= x.ndim:
            # Defer to PyTorch error handling
            return torch.argmax(x, dim=self.dim)

        # If reduction dimension is empty, let PyTorch handle exceptions/behavior
        if x.shape[d] == 0:
            return torch.argmax(x, dim=self.dim)

        # Move reduction dim to the last axis and make K contiguous only once
        xp = x.movedim(d, -1).contiguous()
        B = xp.numel() // xp.shape[-1]
        K = xp.shape[-1]
        x2d = xp.view(B, K)

        # Allocate output
        out_idx = torch.empty(B, dtype=torch.int64, device=x.device)

        if B > 0:
            # Choose a good BLOCK_K at runtime without autotune overhead
            BK = 64
            while BK < K and BK < 1024:
                BK *= 2
            num_warps = 4 if BK <= 256 else 8
            grid = (B,)
            _argmax_row_kernel[grid](
                x2d, out_idx, B, K,
                BLOCK_K=BK,
                num_warps=num_warps,
                num_stages=2,
            )

        # Reshape back to original output shape (with dim d removed)
        out_shape = list(x.shape)
        del out_shape[d]
        return out_idx.view(*out_shape)


batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    # Use CUDA to leverage the Triton kernel; fallback path covers CPU.
    x = torch.randn(batch_size, dim1, dim2, device="cuda")
    return [x]

def get_init_inputs():
    return [1]