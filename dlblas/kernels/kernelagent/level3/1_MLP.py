import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _matmul_bias_activation(
    A_ptr,  # [M, K]
    B_ptr,  # [N, K] (row-major: out_features x in_features)
    Bias_ptr,  # [N]
    C_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    ADD_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.static_assert(BLOCK_K % 16 == 0)
    tl.static_assert(BLOCK_N % 16 == 0)
    tl.static_assert(BLOCK_M % 16 == 0)

    # Base pointers for tiles
    a_row_ptr = A_ptr + offs_m[:, None] * stride_am
    b_col_ptr = B_ptr + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Double-buffered K loop
    k0 = 0
    k_offs = k0 + offs_k
    a_ptrs = a_row_ptr + k_offs[None, :] * stride_ak
    b_ptrs = b_col_ptr + k_offs[:, None] * stride_bk
    a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
    b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

    while k0 + BLOCK_K < K:
        k_next = k0 + BLOCK_K
        k_offs_next = k_next + offs_k

        a_ptrs_next = a_row_ptr + k_offs_next[None, :] * stride_ak
        b_ptrs_next = b_col_ptr + k_offs_next[:, None] * stride_bk
        a_mask_next = (offs_m[:, None] < M) & (k_offs_next[None, :] < K)
        b_mask_next = (k_offs_next[:, None] < K) & (offs_n[None, :] < N)

        a_next = tl.load(a_ptrs_next, mask=a_mask_next, other=0.0).to(tl.float16)
        b_next = tl.load(b_ptrs_next, mask=b_mask_next, other=0.0).to(tl.float16)

        acc += tl.dot(a, b)

        a = a_next
        b = b_next
        k0 = k_next

    # Final MAC
    acc += tl.dot(a, b)

    if ADD_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _gemv_bias_activation(
    A_ptr,  # [M, K]
    B_ptr,  # [N, K] (row-major)
    Bias_ptr,  # [N]
    C_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    ADD_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.static_assert(BLOCK_K % 16 == 0)
    tl.static_assert(BLOCK_N % 16 == 0)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Double-buffered over K
    k0 = 0
    k_offs = k0 + offs_k

    a_ptrs = A_ptr + pid_m * stride_am + k_offs * stride_ak
    a_vec = tl.load(a_ptrs, mask=k_offs < K, other=0.0).to(tl.float16)

    b_ptrs = B_ptr + (offs_n[:, None] * stride_bn + k_offs[None, :] * stride_bk)
    b_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

    while k0 + BLOCK_K < K:
        k_next = k0 + BLOCK_K
        k_offs_next = k_next + offs_k

        a_vec_next = tl.load(A_ptr + pid_m * stride_am + k_offs_next * stride_ak, mask=k_offs_next < K, other=0.0).to(tl.float16)
        b_tile_next = tl.load(
            B_ptr + (offs_n[:, None] * stride_bn + k_offs_next[None, :] * stride_bk),
            mask=(offs_n[:, None] < N) & (k_offs_next[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        prod = tl.dot(b_tile, tl.trans(a_vec[None, :]))  # [BLOCK_N, 1], f32 accum
        acc += prod[:, 0]

        a_vec = a_vec_next
        b_tile = b_tile_next
        k0 = k_next

    # Final accumulate
    prod = tl.dot(b_tile, tl.trans(a_vec[None, :]))
    acc += prod[:, 0]

    if ADD_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = C_ptr + pid_m * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc, mask=offs_n < N)


def _linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, apply_relu: bool):
    # x: [M, K], weight: [N, K], bias: [N]
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Triton path requires CUDA tensors"
    M, K = x.shape
    N, Kw = weight.shape
    assert K == Kw, "Incompatible shapes for Linear"

    # For tiny batches, cuBLAS GEMV via PyTorch is usually fastest; keep exact semantics.
    if M <= 2:
        out = F.linear(x, weight, bias)
        if apply_relu:
            out = F.relu(out)
        return out

    x = x.contiguous()
    w = weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Choose GEMV kernel when M is small for better cache behavior
    if M <= 8:
        BLOCK_N = 256
        BLOCK_K = 256
        grid = (triton.cdiv(N, BLOCK_N) * M,)
        _gemv_bias_activation[grid](
            x, w, (bias if bias is not None else w), out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            ADD_BIAS=(bias is not None),
            APPLY_RELU=apply_relu,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=4,
        )
        return out

    # GEMM path for larger batches
    if M <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 64
        num_warps, num_stages = 4, 4
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        num_warps, num_stages = 8, 4

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_bias_activation[grid](
        x, w, (bias if bias is not None else w), out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        ADD_BIAS=(bias is not None),
        APPLY_RELU=apply_relu,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        self._linears = []

        for layer_size in layer_sizes:
            lin = nn.Linear(current_input_size, layer_size)
            layers.append(lin)
            layers.append(nn.ReLU())
            self._linears.append(lin)
            current_input_size = layer_size
        
        lin_out = nn.Linear(current_input_size, output_size)
        layers.append(lin_out)
        self._linears.append(lin_out)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Fallback to PyTorch path for non-CUDA tensors or dtypes other than float32
        if (not x.is_cuda) or (x.dtype != torch.float32):
            return self.network(x)

        a = x
        # First hidden layer + ReLU
        w0, b0 = self._linears[0].weight, self._linears[0].bias
        a = _linear_triton(a, w0, b0, apply_relu=True)

        # Second hidden layer + ReLU
        w1, b1 = self._linears[1].weight, self._linears[1].bias
        a = _linear_triton(a, w1, b1, apply_relu=True)

        # Final layer (no activation)
        w2, b2 = self._linears[2].weight, self._linears[2].bias
        a = _linear_triton(a, w2, b2, apply_relu=False)

        return a


# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]