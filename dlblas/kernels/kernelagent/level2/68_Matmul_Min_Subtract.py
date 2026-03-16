import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_min_sub_kernel(
    X_ptr, W_ptr, B_ptr, C_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_b,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Outer-product accumulation across K for small K
    k = 0
    while k < K:
        x_ptrs = X_ptr + offs_m * stride_xm + k * stride_xk          # [BM]
        w_ptrs = W_ptr + offs_n * stride_wn + k * stride_wk          # [BN]

        x_vec = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)   # [BM]
        w_vec = tl.load(w_ptrs, mask=mask_n, other=0.0).to(tl.float32)   # [BN]

        acc += x_vec[:, None] * w_vec[None, :]
        k += 1

    # Fuse bias and scalar constant: min(z, c) - c == min(z - c, 0)
    b = tl.load(B_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)  # [BN]
    c = tl.load(C_ptr).to(tl.float32)
    bc = b - c
    out = tl.minimum(acc + bc[None, :], 0.0)

    # Store
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, out, mask=y_mask)


def fused_linear_min_sub(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, constant: torch.Tensor):
    # Shapes
    M, K = x.shape
    N = weight.shape[0]

    # Ensure contiguous tensors
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()
    c_c = constant.contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Heuristics tuned for small shapes (e.g., M=128, K~10, N~5)
    BLOCK_M = 128
    BLOCK_N = 8
    BLOCK_K = 1  # outer-product step

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_linear_min_sub_kernel[grid](
        x_c, w_c, b_c, c_c, y,
        M, N, K,
        x_c.stride(0), x_c.stride(1),
        w_c.stride(0), w_c.stride(1),
        b_c.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=1, num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        # Use Triton path on CUDA when gradients are not required (inference), else fall back to PyTorch.
        use_triton = (
            x.is_cuda
            and not (x.requires_grad
                     or self.linear.weight.requires_grad
                     or (self.linear.bias is not None and self.linear.bias.requires_grad)
                     or self.constant.requires_grad)
        )
        if use_triton:
            return fused_linear_min_sub(
                x,
                self.linear.weight,
                self.linear.bias if self.linear.bias is not None else torch.zeros(self.linear.weight.shape[0], device=x.device, dtype=x.dtype),
                self.constant.to(device=x.device, dtype=x.dtype),
            )
        else:
            y = self.linear(x)
            y = torch.min(y, self.constant)
            y = y - self.constant
            return y


batch_size = 128
in_features = 10
out_features = 5
constant = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, constant]