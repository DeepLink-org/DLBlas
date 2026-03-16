import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small-batch friendly tiles
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 128}, num_warps=1, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_fw_kernel(
    X_ptr,        # [M, K]
    W_ptr,        # [N, K] (nn.Linear.weight)
    B_ptr,        # [N]
    Y_ptr,        # [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_ids = k0 + offs_k

        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + k_ids[None, :] * stride_wk)

        x_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k_ids[None, :] < K)

        # Prefer L2 for streaming, keep math in fp32
        x = tl.load(x_ptrs, mask=x_mask, other=0.0, cache_modifier=".cg")
        w = tl.load(w_ptrs, mask=w_mask, other=0.0, cache_modifier=".cg")

        acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
        k0 += BLOCK_K

    # Add bias: [N] -> [BM, BN]
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _linear_rowwise_small_kernel(
    X_ptr,        # [M, K]
    W_ptr,        # [N, K]
    B_ptr,        # [N]
    Y_ptr,        # [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1 CTA computes the whole output row (all N) for a single m
    pid_m = tl.program_id(axis=0)
    m_mask = pid_m < M

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Bias upfront
    b = tl.load(B_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x = tl.load(
            X_ptr + pid_m * stride_xm + offs_k * stride_xk,
            mask=m_mask & (offs_k < K),
            other=0.0,
            cache_modifier=".cg",
        )
        # Load W chunk as [BK, BN]; W layout is [N, K]
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0, cache_modifier=".cg")

        acc += tl.sum(x[:, None].to(tl.float32) * w.to(tl.float32), axis=0)
        k0 += BLOCK_K

    tl.store(
        Y_ptr + pid_m * stride_ym + offs_n * stride_yn,
        acc,
        mask=m_mask & n_mask,
    )


def _triton_linear(x_2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # x_2d: [M, K], weight: [N, K], bias: [N]
    M, K = x_2d.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x_2d.device, dtype=torch.float32)

    stride_xm, stride_xk = x_2d.stride()
    stride_wn, stride_wk = weight.stride()
    stride_ym, stride_yn = y.stride()

    # Heuristic: for tiny M,N use 1D rowwise kernel to reduce grid/launch overhead.
    if (M <= 32) and (N <= 32):
        BLOCK_N = 16 if N <= 16 else 32
        # Favor larger K chunks to reduce loop iters for small problems
        BLOCK_K = 128 if K >= 128 else 64
        grid_small = (M,)
        _linear_rowwise_small_kernel[grid_small](
            x_2d, weight, bias, y,
            M, N, K,
            stride_xm, stride_xk,
            stride_wn, stride_wk,
            stride_ym, stride_yn,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=1,
            num_stages=2,
        )
        return y

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _linear_fw_kernel[grid](
        x_2d, weight, bias, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(ModelNew, self).__init__()
        # Initialize hidden state with random values
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers, batch_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Preserve original semantics exactly
        self.h0 = self.h0.to(x.device)
        self.c0 = self.h0.to(x.device)
        
        # Forward propagate LSTM
        out, hn = self.lstm(x, (self.h0, self.c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        last = out[:, -1, :]  # [batch_size, hidden_size]

        if last.is_cuda and self.fc.weight.is_cuda and self.fc.bias is not None and self.fc.bias.is_cuda:
            return _triton_linear(last, self.fc.weight, self.fc.bias)
        else:
            return self.fc(last)

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]