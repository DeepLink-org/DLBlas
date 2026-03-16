import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import triton
import triton.language as tl


@triton.jit
def _exp_segsum_from_cumsum_kernel(x_cumsum_ptr, out_ptr, T: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute exp(segsum(x)) given x_cumsum = cumsum(x, dim=-1).
    For each vector v (length T), output O (T x T):
        O[i, j] = exp(v[i] - v[j]) if i >= j else 0
    One program per vector. Uses outer-product trick to reduce exp calls from O(T^2) to O(T).
    """
    pid = tl.program_id(axis=0)

    offs_i = tl.arange(0, BLOCK)
    offs_j = tl.arange(0, BLOCK)
    mask_i = offs_i < T
    mask_j = offs_j < T

    base_in = pid * T
    row_vals = tl.load(x_cumsum_ptr + base_in + offs_i, mask=mask_i, other=0.0)
    col_vals = tl.load(x_cumsum_ptr + base_in + offs_j, mask=mask_j, other=0.0)

    # Compute exponentials once per row/col, then outer-product
    exp_row = tl.exp(row_vals)        # shape: (BLOCK,)
    exp_neg_col = tl.exp(-col_vals)   # shape: (BLOCK,)

    out_tile = exp_row[:, None] * exp_neg_col[None, :]

    # Lower-triangular mask (i >= j) and in-bounds mask
    tri_mask = offs_i[:, None] >= offs_j[None, :]
    inb_mask = mask_i[:, None] & mask_j[None, :]
    out_tile = tl.where(tri_mask & inb_mask, out_tile, 0.0)

    base_out = pid * T * T
    store_offsets = offs_i[:, None] * T + offs_j[None, :]
    tl.store(out_ptr + base_out + store_offsets, out_tile, mask=inb_mask)


def _exp_segsum_from_cumsum(x_cumsum: torch.Tensor) -> torch.Tensor:
    """
    Helper to compute exp(segsum(x)) from a precomputed cumulative sum vector along the last dim.
    x_cumsum: (..., T)
    returns: (..., T, T)
    """
    if not x_cumsum.is_cuda:
        # CPU fallback
        T = x_cumsum.size(-1)
        v = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x_cumsum.device, dtype=torch.bool), diagonal=0)
        v = v.masked_fill(~mask, float('-inf'))
        return torch.exp(v)

    T = x_cumsum.shape[-1]
    lead_shape = x_cumsum.shape[:-1]
    tot = int(x_cumsum.numel() // T)
    x_flat = x_cumsum.contiguous().view(tot, T)
    out = torch.empty((tot, T, T), device=x_cumsum.device, dtype=x_cumsum.dtype)

    # Adaptive block size and warps for occupancy and minimal masking
    if T <= 32:
        BLOCK = 32
        num_warps = 2
    elif T <= 64:
        BLOCK = 64
        num_warps = 4
    else:
        BLOCK = 128
        num_warps = 8

    grid = (tot,)
    _exp_segsum_from_cumsum_kernel[grid](x_flat, out, T=T, BLOCK=BLOCK, num_warps=num_warps, num_stages=2)
    return out.view(*lead_shape, T, T)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def segsum(self, x):
        """Naive segment sum calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        # A_cumsum along intra-chunk time l
        A_blocks_bhcl = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks_bhcl, dim=-1)  # (b, h, c, l)
        
        # 1. Compute diagonal block outputs
        # Compute L = exp(segsum(A_blocks)) efficiently from cumsum with Triton
        # L shape: (b, h, c, l, l), where L[..., i, j] = exp(cumsum[i] - cumsum[j]) if i >= j else 0
        L = _exp_segsum_from_cumsum(A_cumsum)  # (b, h, c, l, l)

        # Re-express the 4-way einsum via two GEMMs + elementwise multiply for better performance:
        # For each (b, c, h):
        #   K[s, l] = sum_n B[s, n] * C[l, n] = (B @ C^T)[s, l]
        #   M[l, s] = L[l, s] * K[s, l]     -> use K^T to match (l, s)
        #   Y[l, p] = (M @ X)[l, p]
        b, c, l, h, p = *X_blocks.shape[:4], X_blocks.shape[-1]
        n = self.d_state

        # Bring tensors into (b, c, h, ...)
        X_bchlp = rearrange(X_blocks, "b c l h p -> b c h l p").contiguous()
        B_bchln = rearrange(B_blocks, "b c l h n -> b c h l n").contiguous()
        C_bchln = rearrange(C_blocks, "b c l h n -> b c h l n").contiguous()
        L_bchls = rearrange(L, "b h c l s -> b c h l s").contiguous()

        BCH = b * c * h
        # Flatten batch dims for bmm
        X_mat = X_bchlp.view(BCH, l, p)                # (BCH, S, P)
        B_mat = B_bchln.view(BCH, l, n)                # (BCH, S, N)
        C_mat = C_bchln.view(BCH, l, n)                # (BCH, L, N)
        L_mat = L_bchls.view(BCH, l, l)                # (BCH, L, S) since L=S=l

        # K = B @ C^T -> (BCH, S, L)
        K = torch.bmm(B_mat, C_mat.transpose(1, 2))
        # M = L .* K^T -> (BCH, L, S)
        M = L_mat * K.transpose(1, 2)
        # Y = M @ X -> (BCH, L, P)
        Y_mat = torch.bmm(M, X_mat)
        # Reshape back to (b, c, l, h, p)
        Y_diag = Y_mat.view(b, c, h, l, p).permute(0, 1, 3, 2, 4).contiguous()

        # 2. Compute intra-chunk states using scaled-B bmm: states = (X^T) @ (diag(decay) @ B)
        # decay_states: (b, h, c, l) -> (b, c, h, l)
        decay_bchl = decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum)).permute(0, 2, 1, 3).contiguous()
        # Scale B along l by decay
        B_scaled = B_bchln * decay_bchl[..., None]               # (b, c, h, l, n)
        X_mat_T = X_mat.transpose(1, 2).contiguous()             # (BCH, p, l)
        states_mat = torch.bmm(X_mat_T, B_scaled.view(BCH, l, n))  # (BCH, p, n)
        states = states_mat.view(b, c, h, p, n)                  # (b, c, h, p, n)
        
        # 3. Compute inter-chunk recurrence via bmm
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)  # (b, c+1, h, p, n)
        
        # Compute decay_chunk = exp(segsum(pad(A_cumsum[..., -1], (1, 0))))
        # Use Triton-accelerated exp(segsum) from cumsum over chunk dimension
        last_acum = A_cumsum[:, :, :, -1]                    # (b, h, c)
        last_acum_padded = F.pad(last_acum, (1, 0))          # (b, h, c+1)
        chunk_cumsum = torch.cumsum(last_acum_padded, dim=-1)  # (b, h, c+1)
        decay_chunk = _exp_segsum_from_cumsum(chunk_cumsum)    # (b, h, c+1, c+1)

        # new_states[z, ...] = sum_c decay_chunk[z, c] * states[c, ...]
        z = states.shape[1]  # c+1
        BH = b * h
        D = decay_chunk.contiguous().view(BH, z, z)  # (BH, z, z)
        S = states.permute(0, 2, 1, 3, 4).contiguous().view(BH, z, p * n)  # (BH, z, pn)
        out_m = torch.bmm(D, S).view(b, h, z, p, n)  # (b, h, z, p, n)
        new_states = out_m.permute(0, 2, 1, 3, 4).contiguous()  # (b, z, h, p, n)

        return new_states[:, -1]


# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]