import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rowwise_linear_sum_kernel(
    x_ptr,           # (B, I)
    wsum_ptr,        # (I,)
    out_ptr,         # (B,) result
    B: tl.constexpr,
    I: tl.constexpr,
    stride_x_b,
    stride_x_i,
    stride_wsum,
    stride_out_b,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_rows = rows < B

    # Accumulator for each row in the block
    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    # Precompute row base pointers for coalesced access
    row_ptrs = x_ptr + rows[:, None] * stride_x_b

    # Use a runtime-controlled loop (no tl.static_range) to avoid constexpr issues.
    # Unroll by 2 to reduce loop overhead while keeping correct masking.
    k_start = 0
    while k_start < I:
        # Iteration 0
        k_idx0 = k_start + tl.arange(0, BLOCK_K)
        mask_k0 = k_idx0 < I
        x_tile0 = tl.load(
            row_ptrs + k_idx0[None, :] * stride_x_i,
            mask=mask_rows[:, None] & mask_k0[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile0 = tl.load(
            wsum_ptr + k_idx0 * stride_wsum,
            mask=mask_k0,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_tile0 * w_tile0[None, :], axis=1)

        # Iteration 1 (may be fully masked if beyond I)
        k_idx1 = k_start + BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k1 = k_idx1 < I
        x_tile1 = tl.load(
            row_ptrs + k_idx1[None, :] * stride_x_i,
            mask=mask_rows[:, None] & mask_k1[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile1 = tl.load(
            wsum_ptr + k_idx1 * stride_wsum,
            mask=mask_k1,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_tile1 * w_tile1[None, :], axis=1)

        k_start += 2 * BLOCK_K

    # Write result
    tl.store(out_ptr + rows * stride_out_b, acc, mask=mask_rows)


@triton.jit
def _fused_linear_sum_kernel(
    x_ptr,            # *f32 (B, I)
    W_ptr,            # *f32 (O, I)
    b_ptr,            # *f32 (O,) - can be dummy if O_b==0
    out_ptr,          # *f32 (B,)
    B, I, O,          # int32 sizes
    stride_x_b,       # int32
    stride_x_i,       # int32
    stride_w_o,       # int32
    stride_w_i,       # int32
    stride_b_o,       # int32
    stride_out_b,     # int32
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL_O: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_rows = rows < B

    # Base ptrs
    row_ptrs = x_ptr + rows[:, None] * stride_x_b

    # Accumulator per row
    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    # Precompute bias sum once per program
    c_acc = tl.zeros((), dtype=tl.float32)
    offs_o = tl.arange(0, UNROLL_O)
    o = 0
    while o < O:
        o_idx = o + offs_o
        mask_o = o_idx < O
        b_vals = tl.load(b_ptr + o_idx * stride_b_o, mask=mask_o, other=0.0).to(tl.float32)
        c_acc += tl.sum(b_vals, axis=0)
        o += UNROLL_O

    # Iterate over I in tiles
    offs_k = tl.arange(0, BLOCK_K)
    k = 0
    while k < I:
        k_idx = k + offs_k
        mask_k = k_idx < I

        # Compute weight column-sum tile v[k] = sum_o W[o, k]
        v_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
        o2 = 0
        while o2 < O:
            o2_idx = o2 + offs_o
            mask_o2 = o2_idx < O
            w_block = tl.load(
                W_ptr + o2_idx[:, None] * stride_w_o + k_idx[None, :] * stride_w_i,
                mask=mask_o2[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            v_tile += tl.sum(w_block, axis=0)
            o2 += UNROLL_O

        # Load x tile and accumulate dot for all rows in this block
        x_tile = tl.load(
            row_ptrs + k_idx[None, :] * stride_x_i,
            mask=mask_rows[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x_tile * v_tile[None, :], axis=1)

        k += BLOCK_K

    # Add bias sum and store
    acc += c_acc
    tl.store(out_ptr + rows * stride_out_b, acc, mask=mask_rows)


class ModelNew(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Fallback to PyTorch path when not on CUDA or when autograd is required
        if (not x.is_cuda) or x.requires_grad:
            y = self.linear(x)  # (batch_size, out_features)
            y = torch.sum(y, dim=1, keepdim=True)  # (batch_size, 1)
            y = torch.max(y, dim=1, keepdim=True)[0]  # (batch_size, 1)
            y = torch.mean(y, dim=1, keepdim=True)  # (batch_size, 1)
            y = torch.logsumexp(y, dim=1, keepdim=True)  # (batch_size, 1)
            y = torch.logsumexp(y, dim=1, keepdim=True)  # (batch_size, 1)
            return y

        # Optimized CUDA path using Triton:
        # The chain reduces to:
        # sum_j (x @ W^T + b)_j = x @ (sum_j W_j)^T + sum_j b_j
        # Fuse the computation of sum_j W_j and sum_j b_j inside the kernel
        B, I = x.shape

        # Ensure contiguous tensors
        x_c = x.contiguous()
        W = self.linear.weight.contiguous()        # (O, I)
        b = self.linear.bias
        if b is None:
            b_c = torch.empty(1, device=x.device, dtype=W.dtype)  # dummy; O_b=0 prevents use
            O_b = 0
        else:
            b_c = b.contiguous()
            O_b = W.shape[0]

        # Output buffer (B, 1) but we store as (B,) in kernel and then view
        out = torch.empty((B,), device=x.device, dtype=torch.float32)

        # Launch Triton kernel: process BLOCK_B rows per program
        BLOCK_B = 256
        BLOCK_K = 32
        UNROLL_O = 32
        grid = (triton.cdiv(B, BLOCK_B),)

        _fused_linear_sum_kernel[grid](
            x_c, W, b_c, out,
            B, I, O_b,
            x_c.stride(0), x_c.stride(1),
            W.stride(0), W.stride(1),
            b_c.stride(0),
            out.stride(0),
            BLOCK_B=BLOCK_B,
            BLOCK_K=BLOCK_K,
            UNROLL_O=UNROLL_O,
            num_warps=4,
            num_stages=2,
        )

        return out.view(B, 1)


batch_size = 128
in_features = 10
out_features = 5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]