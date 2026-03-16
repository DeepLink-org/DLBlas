import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_linear_sub_mul_relu_kernel(
    A_ptr,  # [M, K]
    W_ptr,  # [N, K] but accessed as [K, N] using strides
    B_ptr,  # [N]
    C_ptr,  # [M, N]
    SUB_VAL: tl.constexpr,  # scalar subtraction value
    MUL_VAL: tl.constexpr,  # scalar multiplication value
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,  # treat weight as [K, N] with these strides
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        a_mask = (mask_m[:, None]) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (mask_n[None, :])

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, w)

    # Bias add
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Epilogue: (acc - SUB_VAL) * MUL_VAL then ReLU
    acc = (acc - SUB_VAL) * MUL_VAL
    acc = tl.maximum(acc, 0.0)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        # Triton-accelerated fused path for CUDA float32 tensors
        use_triton = (
            x.is_cuda
            and x.dtype == torch.float32
            and x.ndim == 2
            and x.shape[1] == self.linear.weight.shape[1]
            and self.linear.weight.is_cuda
            and (self.linear.bias is not None and self.linear.bias.is_cuda)
        )
        if use_triton:
            W = self.linear.weight  # [out_features, in_features] == [N, K]
            B = self.linear.bias    # [N]
            M, K = x.shape
            N = W.shape[0]

            # Allocate output
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)

            # Strides
            stride_am, stride_ak = x.stride(0), x.stride(1)
            # Treat W as [K, N] via strides (no actual transpose)
            stride_wk, stride_wn = W.stride(1), W.stride(0)
            stride_cm, stride_cn = out.stride(0), out.stride(1)

            # Tile sizes tuned for small N,K; keep BLOCK_* >= 16 to satisfy tl.dot constraints
            BLOCK_M = 128
            BLOCK_N = 32
            BLOCK_K = 32

            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _fused_linear_sub_mul_relu_kernel[grid](
                x, W, B, out,
                self.subtract_value, self.multiply_value,
                M, N, K,
                stride_am, stride_ak,
                stride_wk, stride_wn,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
            return out

        # Fallback to PyTorch reference for other cases
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = torch.relu(x)
        return x


batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]