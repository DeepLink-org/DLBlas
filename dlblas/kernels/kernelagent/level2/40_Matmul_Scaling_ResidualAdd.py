import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_fused_kernel(
    A_ptr,         # [M, K]
    WT_ptr,        # we pass W in [N, K] here (keep name for signature compatibility)
    B_ptr,         # [N]
    Y_ptr,         # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,  # stride_wk: stride along K of W, stride_wn: stride along N of W
    stride_ym, stride_yn,
    scale,         # fused scale = 1 + scaling_factor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        # Use W in row-major [N, K] to avoid a separate transpose on the host
        w_ptrs = WT_ptr + (offs_n[:, None] * stride_wn + (k0 + offs_k[None, :]) * stride_wk)

        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k0 + offs_k[None, :] < K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)           # (BM, BK)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)           # (BN, BK)

        # acc += a @ w^T
        acc += tl.dot(a, tl.trans(w))
        k0 += BLOCK_K

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Fused scaling + residual: y = (1 + s) * (A @ W^T + b)
    acc *= scale

    # Store
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        if x.is_cuda:
            A = x.contiguous()  # [M, K]
            W = self.matmul.weight.contiguous()  # [N, K] row-major; avoid transpose overhead
            b = self.matmul.bias

            M, K = A.shape
            N = W.shape[0]
            # Prepare bias: handle None gracefully
            if b is None:
                b_buf = torch.zeros(N, device=x.device, dtype=A.dtype)
            else:
                b_buf = b.contiguous()

            Y = torch.empty((M, N), device=x.device, dtype=torch.float32)

            # Grid setup
            def grid(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_M"]),
                    triton.cdiv(N, meta["BLOCK_N"]),
                )

            # Use tile sizes that provide 4 CTAs for 128x128 and good occupancy
            _linear_fused_kernel[grid](
                A, W, b_buf, Y,
                M, N, K,
                A.stride(0), A.stride(1),
                W.stride(1), W.stride(0),  # pass (stride_wk, stride_wn)
                Y.stride(0), Y.stride(1),
                1.0 + float(self.scaling_factor),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                num_warps=8, num_stages=3,
            )
            return Y
        else:
            # CPU fallback: exact original semantics
            x = self.matmul(x)
            original_x = x.clone().detach()
            x = x * self.scaling_factor
            x = x + original_x
            return x


batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]