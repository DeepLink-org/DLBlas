import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_last_step_kernel(
    A_ptr,  # [B, K]
    W_ptr,  # [O, K]
    Bias_ptr,  # [O]
    C_ptr,  # [B, O]
    B, K, O,
    stride_a0, stride_a1,
    stride_w0, stride_w1,
    stride_c0, stride_c1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # along batch B
    pid_n = tl.program_id(axis=1)  # along output O

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                     # [BK]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_offsets = k_iter + offs_k

        # A tile: [BM, BK] => A[m, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_a0 + k_offsets[None, :] * stride_a1)
        a_mask = (offs_m[:, None] < B) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # W tile laid out as [BK, BN] => W[n, k]
        w_ptrs = W_ptr + (offs_n[None, :] * stride_w0 + k_offsets[:, None] * stride_w1)
        w_mask = (offs_n[None, :] < O) & (k_offsets[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, w)
        k_iter += BLOCK_K

    # Add bias
    b = tl.load(Bias_ptr + offs_n, mask=offs_n < O, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1)
    c_mask = (offs_m[:, None] < B) & (offs_n[None, :] < O)
    tl.store(c_ptrs, acc, mask=c_mask)


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
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Match original semantics: move h0 to device and set c0 equal to h0 on that device
        self.h0 = self.h0.to(x.device)
        self.c0 = self.h0.to(x.device)
        
        # Forward propagate LSTM
        out, hn = self.lstm(x, (self.h0, self.c0))  # out: [B, S, 2*hidden]
        
        # Decode the hidden state of the last time step
        last = out[:, -1, :]  # [B, K], K = 2*hidden
        
        # Triton-accelerated final linear layer on CUDA float32; otherwise fallback
        if last.is_cuda and self.fc.weight.is_cuda and last.dtype == torch.float32 and self.fc.weight.dtype == torch.float32:
            B = last.shape[0]
            K = last.shape[1]
            O = self.fc.weight.shape[0]

            a = last
            w = self.fc.weight  # [O, K]
            b = self.fc.bias if self.fc.bias is not None else torch.zeros(O, device=last.device, dtype=last.dtype)

            out_linear = torch.empty((B, O), device=last.device, dtype=torch.float32)

            # Tuned tile sizes for small B/O and moderate K
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 128
            grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(O, BLOCK_N))
            _linear_last_step_kernel[grid](
                a, w, b, out_linear,
                B, K, O,
                a.stride(0), a.stride(1),
                w.stride(0), w.stride(1),
                out_linear.stride(0), out_linear.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )
            out = out_linear
        else:
            out = self.fc(last)  # [B, O]
        
        return out


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