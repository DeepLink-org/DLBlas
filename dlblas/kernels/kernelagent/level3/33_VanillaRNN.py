import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _i2h_tanh_kernel(
    x_ptr,           # float32[B, I]
    hprev_ptr,       # float32[B, H]
    w1_ptr,          # float32[H, I+H] -> row-major (out, in)
    b1_ptr,          # float32[H]
    hout_ptr,        # float32[B, H]
    B, I, H, K,      # ints: K = I + H
    BLOCK_M: tl.constexpr,  # tile along batch
    BLOCK_N: tl.constexpr,  # tile along hidden/out dim
    BLOCK_K: tl.constexpr,  # tile along reduction
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_broadcast = offs_m[:, None]
    offs_n_broadcast = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K = I + H
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_broadcast_row = offs_k[None, :]
        offs_k_broadcast_col = offs_k[:, None]

        # Load concatenated input (x | hprev) without explicitly concatenating
        mask_rows = offs_m_broadcast < B

        # part from x: valid when offs_k < I
        mask_x = (offs_k_broadcast_row < I) & mask_rows
        px = x_ptr + offs_m_broadcast * I + offs_k_broadcast_row
        # Hint contiguity for better vectorization
        x_vals = tl.load(px, mask=mask_x, other=0.0)

        # part from hprev: valid when I <= offs_k < I+H
        mask_h = (offs_k_broadcast_row >= I) & (offs_k_broadcast_row < (I + H)) & mask_rows
        ph = hprev_ptr + offs_m_broadcast * H + (offs_k_broadcast_row - I)
        h_vals = tl.load(ph, mask=mask_h, other=0.0)

        # Only one of masks is true per column, so sum is a select
        x_tile = x_vals + h_vals

        # Load weight tile transposed for dot: want (BLOCK_K, BLOCK_N)
        mask_w = (offs_n_broadcast < H) & (offs_k_broadcast_col < K)
        pw = w1_ptr + offs_n_broadcast * K + offs_k_broadcast_col
        w_tile_t = tl.load(pw, mask=mask_w, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile_t)

    # Add bias and tanh
    b = tl.load(b1_ptr + offs_n, mask=offs_n < H, other=0.0)
    acc += b[None, :]
    # tanh(x) = 2 * sigmoid(2x) - 1
    acc = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * acc))) - 1.0

    # Store output hidden tile
    mask_store = (offs_m_broadcast < B) & (offs_n_broadcast < H)
    pout = hout_ptr + offs_m_broadcast * H + offs_n_broadcast
    tl.store(pout, acc, mask=mask_store)


@triton.jit
def _linear_bias_kernel(
    inp_ptr,     # float32[B, K]
    w_ptr,       # float32[O, K] row-major (out, in)
    b_ptr,       # float32[O]
    out_ptr,     # float32[B, O]
    B, K, O,     # ints
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_broadcast = offs_m[:, None]
    offs_n_broadcast = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        offs_k_row = offs_k[None, :]
        offs_k_col = offs_k[:, None]

        # Load input tile (B, Ktile)
        pa = inp_ptr + offs_m_broadcast * K + offs_k_row
        mask_a = (offs_m_broadcast < B) & (offs_k_row < K)
        a_tile = tl.load(pa, mask=mask_a, other=0.0)

        # Load weight tile as (Ktile, Ntile)
        pw = w_ptr + offs_n_broadcast * K + offs_k_col
        mask_w = (offs_n_broadcast < O) & (offs_k_col < K)
        w_tile_t = tl.load(pw, mask=mask_w, other=0.0)

        acc += tl.dot(a_tile, w_tile_t)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < O, other=0.0)
    acc += b[None, :]

    # Store
    mask_store = (offs_m_broadcast < B) & (offs_n_broadcast < O)
    pout = out_ptr + offs_m_broadcast * O + offs_n_broadcast
    tl.store(pout, acc, mask=mask_store)


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
        :return: Output tensor of shape (batch_size, output_size).
        """
        # Ensure hidden state is on the same device as input
        self.hidden = self.hidden.to(x.device)

        # Fallback to PyTorch if not CUDA
        if not x.is_cuda:
            combined = torch.cat((x, self.hidden), dim=1)
            self.hidden = self.tanh(self.i2h(combined))
            output = self.h2o(self.hidden)
            return output

        # Prepare tensors
        B = x.shape[0]
        I = self.input_size
        H = self.hidden_size
        O = self.output_size
        K1 = I + H

        x_contig = x.contiguous()
        hprev_contig = self.hidden.contiguous()
        # Ensure parameters are on the same device as input for Triton kernels
        w1 = self.i2h.weight.to(x.device).contiguous()
        b1 = self.i2h.bias.to(x.device).contiguous()
        w2 = self.h2o.weight.to(x.device).contiguous()
        b2 = self.h2o.bias.to(x.device).contiguous()

        # Allocate outputs
        hidden_out = torch.empty((B, H), device=x.device, dtype=x.dtype)
        output = torch.empty((B, O), device=x.device, dtype=x.dtype)

        # Launch Triton kernel for hidden = tanh([x|hprev] @ W1^T + b1)
        # Keep >=16 for tl.dot
        BLOCK_M1 = 16
        BLOCK_N1 = 32
        BLOCK_K1 = 256
        grid1 = (triton.cdiv(B, BLOCK_M1), triton.cdiv(H, BLOCK_N1))
        _i2h_tanh_kernel[grid1](
            x_contig, hprev_contig, w1, b1, hidden_out,
            B, I, H, K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=8, num_stages=4
        )

        # Update hidden state
        self.hidden = hidden_out

        # Launch Triton kernel for output = hidden @ W2^T + b2
        BLOCK_M2 = 16
        BLOCK_N2 = 32
        BLOCK_K2 = 128
        grid2 = (triton.cdiv(B, BLOCK_M2), triton.cdiv(O, BLOCK_N2))
        _linear_bias_kernel[grid2](
            self.hidden, w2, b2, output,
            B, H, O,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=4
        )

        return output


batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]