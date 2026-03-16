import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_i2h_tanh_kernel(
    x_ptr,             # float*  [B, I]
    h_prev_ptr,        # float*  [B, H_IN]
    w_ptr,             # float*  [O, I + H_IN]
    b_ptr,             # float*  [O]
    out_ptr,           # float*  [B, O]
    stride_xb, stride_xi,
    stride_hb, stride_hi,
    stride_wo, stride_wi,
    stride_out_b, stride_out_o,
    B: tl.constexpr,                 # batch size
    I: tl.constexpr,                 # input size
    H_IN: tl.constexpr,              # hidden input size
    O: tl.constexpr,                 # output size (hidden_size for i2h)
    BLOCK_B: tl.constexpr,           # tile size in batch dimension
    BLOCK_O: tl.constexpr,           # tile size in output (hidden) dimension
    BLOCK_K: tl.constexpr,           # tile size in reduction dimension
):
    pid_o = tl.program_id(0)
    pid_b = tl.program_id(1)

    o_offsets = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    o_mask = o_offsets < O
    b_mask = b_offsets < B

    acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)

    # Precompute row/col bases to reduce index arithmetic
    x_row_base = b_offsets[:, None] * stride_xb
    h_row_base = b_offsets[:, None] * stride_hb
    w_col_base = o_offsets[None, :] * stride_wo

    # Reduce over input features [0, I) with compile-time unrolling
    for k in tl.static_range(0, I, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < I

        x = tl.load(
            x_ptr + x_row_base + k_offsets[None, :] * stride_xi,
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
        )
        w_t = tl.load(
            w_ptr + w_col_base + k_offsets[:, None] * stride_wi,
            mask=k_mask[:, None] & o_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
        )
        acc += tl.dot(x.to(tl.float32), w_t.to(tl.float32))

    # Reduce over hidden features [I, I+H_IN)
    for k in tl.static_range(0, H_IN, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H_IN

        h_prev = tl.load(
            h_prev_ptr + h_row_base + k_offsets[None, :] * stride_hi,
            mask=b_mask[:, None] & k_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
        )
        w_t = tl.load(
            w_ptr + w_col_base + (I + k_offsets)[:, None] * stride_wi,
            mask=k_mask[:, None] & o_mask[None, :],
            other=0.0,
            cache_modifier=".ca",
        )
        acc += tl.dot(h_prev.to(tl.float32), w_t.to(tl.float32))

    # Add bias
    bias = tl.load(b_ptr + o_offsets, mask=o_mask, other=0.0)
    acc = acc + bias[None, :]

    # tanh(x) = 2 / (1 + exp(-2x)) - 1
    acc = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0

    tl.store(
        out_ptr + b_offsets[:, None] * stride_out_b + o_offsets[None, :] * stride_out_o,
        acc,
        mask=b_mask[:, None] & o_mask[None, :],
    )


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size), and the new hidden state.
        """
        # Ensure hidden is on the same device as input
        self.hidden = self.hidden.to(x.device)

        # If on CUDA, use fused Triton kernel for i2h + tanh; otherwise fall back to PyTorch
        if x.is_cuda and self.i2h.weight.is_cuda and self.i2h.bias.is_cuda and x.dtype == torch.float32:
            B = x.shape[0]
            I = self.input_size
            H_IN = self.hidden_size  # previous hidden size
            O = self.hidden_size     # output hidden size for i2h

            # Prepare tensors
            x_in = x.contiguous()
            h_prev = self.hidden.contiguous()
            w = self.i2h.weight.contiguous()  # [O, I + H_IN]
            b = self.i2h.bias.contiguous()    # [O]
            h_out = torch.empty((B, O), device=x.device, dtype=x.dtype)

            # Tile settings - ensure tl.dot requirements (>=16) are met
            BLOCK_B = 16
            BLOCK_O = 128
            BLOCK_K = 64

            grid = (
                triton.cdiv(O, BLOCK_O),
                triton.cdiv(B, BLOCK_B),
            )
            _fused_i2h_tanh_kernel[grid](
                x_in, h_prev, w, b, h_out,
                x_in.stride(0), x_in.stride(1),
                h_prev.stride(0), h_prev.stride(1),
                w.stride(0), w.stride(1),
                h_out.stride(0), h_out.stride(1),
                B=B, I=I, H_IN=H_IN, O=O,
                BLOCK_B=BLOCK_B, BLOCK_O=BLOCK_O, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            self.hidden = h_out
        else:
            combined = torch.cat((x, self.hidden), dim=1)  # Concatenate input and hidden state
            self.hidden = self.tanh(self.i2h(combined))  # Update hidden state

        # Compute output (kept to preserve original behavior, though not returned)
        _ = self.h2o(self.hidden)
        return self.hidden

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]