import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _gemv_bias_relu_kernel(
    x_ptr,           # (B, IN)
    w_ptr,           # (OUT, IN)
    b_ptr,           # (OUT,)
    y_ptr,           # (B, OUT)
    B, IN_FEATURES, OUT_FEATURES,
    stride_x0, stride_x1,
    stride_w0, stride_w1,
    stride_y0, stride_y1,
    APPLY_RELU: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_o = tl.program_id(axis=1)

    offs_o = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_o = offs_o < OUT_FEATURES

    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    # Loop over input features
    for k in range(0, IN_FEATURES, BLOCK_IN):
        offs_i = k + tl.arange(0, BLOCK_IN)
        mask_i = offs_i < IN_FEATURES

        # Load a tile of x (vector of size BLOCK_IN)
        x = tl.load(
            x_ptr + pid_b * stride_x0 + offs_i * stride_x1,
            mask=mask_i,
            other=0.0,
        ).to(tl.float32)

        # Load a tile of w (BLOCK_OUT x BLOCK_IN)
        w = tl.load(
            w_ptr + offs_o[:, None] * stride_w0 + offs_i[None, :] * stride_w1,
            mask=mask_o[:, None] & mask_i[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate partial dot products
        acc += tl.sum(w * x[None, :], axis=1)

    # Add bias
    b = tl.load(b_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    acc += b

    # Optional ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store
    tl.store(
        y_ptr + pid_b * stride_y0 + offs_o * stride_y1,
        acc,
        mask=mask_o,
    )


@triton.jit
def _mlp_fused_kernel(
    x_ptr,                  # *f32 [B, K0]
    y_ptr,                  # *f32 [B, M_last]
    w_packed_ptr,           # *f32 [sum_l M_l*K_l]
    b_packed_ptr,           # *f32 [sum_l M_l]
    dims_in_ptr,            # *i32 [L]
    dims_out_ptr,           # *i32 [L]
    w_offsets_ptr,          # *i32 [L] offsets in elements
    b_offsets_ptr,          # *i32 [L]
    scratch0_ptr,           # *f32 [B, max_hidden]
    scratch1_ptr,           # *f32 [B, max_hidden]
    B,                      # int
    stride_x0, stride_x1,   # strides for x
    stride_y0, stride_y1,   # strides for y
    scratch_stride0,        # stride for scratch leading dim
    L: tl.constexpr,        # number of layers (constexpr to unroll)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    # Base pointers for this batch element
    x_base = x_ptr + pid_b * stride_x0
    y_base = y_ptr + pid_b * stride_y0

    # Register buffer to carry activations across layers (size BLOCK_N covers all M by contract)
    prev_reg = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Current input vector pointer and stride (elements) for the first layer
    curr_ptr = x_base
    curr_s1 = stride_x1

    for li in tl.static_range(0, L):
        # Load metadata scalars for this layer
        K = tl.load(dims_in_ptr + li)
        M = tl.load(dims_out_ptr + li)
        woff_li = tl.load(w_offsets_ptr + li)
        boff_li = tl.load(b_offsets_ptr + li)
        w_layer = w_packed_ptr + woff_li
        b_layer = b_packed_ptr + boff_li

        offs_o = tl.arange(0, BLOCK_N)
        mask_o = offs_o < M
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        if li == 0:
            # First layer: K can be large; stream from global memory
            k = 0
            while k < K:
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K

                x_vec = tl.load(curr_ptr + offs_k * curr_s1, mask=mask_k, other=0.0)
                # W[row=j, col=k] with row-major [M, K]
                w_ptrs = w_layer + offs_o[:, None] * K + offs_k[None, :]
                w = tl.load(w_ptrs, mask=mask_o[:, None] & mask_k[None, :], other=0.0)

                acc += tl.sum(w * x_vec[None, :], axis=1)
                k += BLOCK_K
        else:
            # Subsequent layers: K == previous M <= BLOCK_N; use register-resident prev_reg
            offs_k = tl.arange(0, BLOCK_N)
            mask_k = offs_k < K
            x_vec = tl.where(mask_k, prev_reg, 0.0)
            w_ptrs = w_layer + offs_o[:, None] * K + offs_k[None, :]
            w = tl.load(w_ptrs, mask=mask_o[:, None] & mask_k[None, :], other=0.0)
            acc += tl.sum(w * x_vec[None, :], axis=1)

        # Add bias
        b = tl.load(b_layer + offs_o, mask=mask_o, other=0.0)
        acc = acc + b

        # Apply ReLU on all but last layer and keep in registers for the next layer
        if li != (L - 1):
            acc = tl.maximum(acc, 0.0)
            prev_reg = acc
        else:
            # Store final output
            tl.store(y_base + offs_o * stride_y1, acc, mask=mask_o)


def _triton_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, apply_relu: bool):
    # Fallback path if not CUDA tensors
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        out = F.linear(x, weight, bias)
        return F.relu(out) if apply_relu else out

    # Ensure 2D shapes and dtype float32
    assert x.dim() == 2 and weight.dim() == 2 and bias.dim() == 1
    B, IN = x.shape
    OUT, INw = weight.shape
    assert INw == IN and bias.numel() == OUT

    # Make strides/contiguity predictable (not strictly required but helps)
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    y = torch.empty((B, OUT), device=x.device, dtype=torch.float32)

    BLOCK_OUT = 64
    BLOCK_IN = 128

    grid = (B, triton.cdiv(OUT, BLOCK_OUT))

    _gemv_bias_relu_kernel[grid](
        x_c, w_c, b_c, y,
        B, IN, OUT,
        x_c.stride(0), x_c.stride(1),
        w_c.stride(0), w_c.stride(1),
        y.stride(0), y.stride(1),
        APPLY_RELU=apply_relu,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_IN=BLOCK_IN,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()

        layers = []
        current_input_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

        # Collect Linear layers for Triton fast path
        self._linears = [m for m in self.network if isinstance(m, nn.Linear)]
        self._L = len(self._linears)
        self._dims_in_list = [lin.in_features for lin in self._linears]
        self._dims_out_list = [lin.out_features for lin in self._linears]
        self._max_hidden = 0 if self._L <= 1 else max(self._dims_out_list[:-1])

        # Pre-pack weights and biases into contiguous buffers (row-major [M, K] as in PyTorch)
        w_tensors = [lin.weight.detach().contiguous().view(-1) for lin in self._linears]
        b_tensors = [lin.bias.detach().contiguous().view(-1) for lin in self._linears]
        w_offsets = [0]
        b_offsets = [0]
        for i in range(1, self._L):
            w_offsets.append(w_offsets[-1] + self._dims_in_list[i - 1] * self._dims_out_list[i - 1])
            b_offsets.append(b_offsets[-1] + self._dims_out_list[i - 1])

        if self._L > 0:
            w_cat = torch.cat(w_tensors, dim=0).to(torch.float32)
            b_cat = torch.cat(b_tensors, dim=0).to(torch.float32)
            self.register_buffer("_w_packed", w_cat, persistent=False)
            self.register_buffer("_b_packed", b_cat, persistent=False)
            self.register_buffer("_dims_in", torch.tensor(self._dims_in_list, dtype=torch.int32), persistent=False)
            self.register_buffer("_dims_out", torch.tensor(self._dims_out_list, dtype=torch.int32), persistent=False)
            self.register_buffer("_w_offsets", torch.tensor(w_offsets, dtype=torch.int32), persistent=False)
            self.register_buffer("_b_offsets", torch.tensor(b_offsets, dtype=torch.int32), persistent=False)
        else:
            self.register_buffer("_w_packed", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("_b_packed", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("_dims_in", torch.empty(0, dtype=torch.int32), persistent=False)
            self.register_buffer("_dims_out", torch.empty(0, dtype=torch.int32), persistent=False)
            self.register_buffer("_w_offsets", torch.empty(0, dtype=torch.int32), persistent=False)
            self.register_buffer("_b_offsets", torch.empty(0, dtype=torch.int32), persistent=False)

        # Reusable scratch buffers for fused kernel (kept for API compatibility; not used by the kernel now)
        max_h = max(1, self._max_hidden)
        self.register_buffer("_scratch0", torch.empty((1, max_h), dtype=torch.float32), persistent=False)
        self.register_buffer("_scratch1", torch.empty((1, max_h), dtype=torch.float32), persistent=False)

        # Triton tuning constants
        self._BLOCK_N = 64
        self._BLOCK_K = 512
        self._num_warps_fused = 4
        self._num_stages_fused = 3

    def _maybe_resize_scratch(self, B: int, device, dtype):
        max_h = max(1, self._max_hidden)
        need_resize = (
            self._scratch0.device != device
            or self._scratch0.dtype != dtype
            or self._scratch0.shape[0] < B
            or self._scratch0.shape[1] < max_h
        )
        if need_resize:
            self._scratch0 = torch.empty((B, max_h), device=device, dtype=dtype)
            self._scratch1 = torch.empty((B, max_h), device=device, dtype=dtype)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Fallback to reference path for unsupported cases
        if self._L == 0:
            return self.network(x)

        use_triton = (
            x.is_cuda
            and x.dtype == torch.float32
            and all(m <= self._BLOCK_N for m in self._dims_out_list)  # fused kernel requires BLOCK_N cover M
        )

        if not use_triton:
            # Generic fallback: try per-layer Triton GEMV, else PyTorch
            if x.is_cuda and x.dtype == torch.float32:
                out = x
                i = 0
                n = len(self.network)
                while i < n:
                    m = self.network[i]
                    if isinstance(m, nn.Linear):
                        apply_relu = (i + 1 < n) and isinstance(self.network[i + 1], nn.ReLU)
                        out = _triton_linear_relu(out, m.weight, m.bias, apply_relu)
                        i += 2 if apply_relu else 1
                    else:
                        out = m(out)
                        i += 1
                return out
            return self.network(x)

        # Ensure packed buffers are on the correct device
        if self._w_packed.device != x.device:
            self._w_packed = self._w_packed.to(x.device)
            self._b_packed = self._b_packed.to(x.device)
            self._dims_in = self._dims_in.to(x.device)
            self._dims_out = self._dims_out.to(x.device)
            self._w_offsets = self._w_offsets.to(x.device)
            self._b_offsets = self._b_offsets.to(x.device)

        B, K0 = x.shape
        assert K0 == self._dims_in_list[0], "Input feature size mismatch"

        # Prepare output and scratch (scratch retained for compatibility)
        y = torch.empty((B, self._dims_out_list[-1]), device=x.device, dtype=x.dtype)
        self._maybe_resize_scratch(B, x.device, x.dtype)

        # Launch one persistent CTA per batch row to process all layers
        grid = (B,)
        _mlp_fused_kernel[grid](
            x, y,
            self._w_packed, self._b_packed,
            self._dims_in, self._dims_out,
            self._w_offsets, self._b_offsets,
            self._scratch0, self._scratch1,
            B,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            self._scratch0.stride(0),
            L=self._L,
            BLOCK_N=self._BLOCK_N,
            BLOCK_K=self._BLOCK_K,
            num_warps=self._num_warps_fused,
            num_stages=self._num_stages_fused,
        )
        return y


# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]