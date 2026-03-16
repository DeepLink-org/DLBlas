import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _deconv1d_stride1_kernel(
    x_ptr,         # *f32/f16/bf16 [B, C_IN, L_IN]
    w_ptr,         # *f32/f16/bf16 [C_IN, C_OUT, K]
    b_ptr,         # *f32[C_OUT] or dummy if no bias
    y_ptr,         # *dtype(x)[B, C_OUT, L_OUT]
    B: tl.constexpr,
    C_IN,
    C_OUT,
    L_IN,
    K,
    L_OUT,
    BLOCK_T: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_t = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c_offsets = tl.arange(0, BLOCK_C)

    # Masks independent of inner loops
    oc_mask = oc_offsets < C_OUT
    t_mask = t_offsets < L_OUT

    # Batch offsets (assuming contiguous tensors)
    x_batch_offset = pid_b * (C_IN * L_IN)
    y_batch_offset = pid_b * (C_OUT * L_OUT)

    # Accumulator [BLOCK_T, BLOCK_OC] in fp32
    acc = tl.zeros((BLOCK_T, BLOCK_OC), dtype=tl.float32)

    # Iterate over input channels in tiles
    c_start = 0
    while c_start < C_IN:
        cin_idx = c_start + c_offsets
        c_mask = cin_idx < C_IN

        # Precompute base ptrs and masks for this C-tile
        # x_base_ptrs refers to input at t_offsets (k=0), then we shift by -k each iteration
        x_base_ptrs = x_ptr + x_batch_offset + cin_idx[:, None] * L_IN + t_offsets[None, :]
        # Weight base for k=0; then add +k each iteration
        w_base_ptrs = w_ptr + (cin_idx[:, None] * C_OUT + oc_offsets[None, :]) * K
        w_mask = c_mask[:, None] & oc_mask[None, :]

        # Iterate over kernel taps
        k_idx = 0
        while k_idx < K:
            # Valid input time positions for this k
            t_in = t_offsets - k_idx
            t_in_mask = (t_in >= 0) & (t_in < L_IN)

            # Addresses for this k
            x_ptrs = x_base_ptrs - k_idx
            x_mask = c_mask[:, None] & t_in_mask[None, :]

            # Loads in original dtype to enable Tensor Cores on fp16/bf16
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0, cache_modifier=".cg")         # [BLOCK_C, BLOCK_T]
            w_tile = tl.load(w_base_ptrs + k_idx, mask=w_mask, other=0.0, cache_modifier=".ca")  # [BLOCK_C, BLOCK_OC]

            # acc += X_tile.T @ W_tile
            acc += tl.dot(tl.trans(x_tile), w_tile)

            k_idx += 1

        c_start += BLOCK_C

    # Add bias if present
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
        acc = acc + b_vals[None, :]

    # Store results in output dtype (same as input's dtype)
    y_ptrs = y_ptr + y_batch_offset + oc_offsets[None, :] * L_OUT + t_offsets[:, None]
    store_mask = t_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Conditions for fast Triton kernel: stride=1, padding=0, output_padding=0, groups=1
        # Fall back to PyTorch for other cases.
        mod = self.conv1d_transpose
        try:
            stride_ok = (mod.stride == (1,) or mod.stride == 1)
            padding_ok = (mod.padding == (0,) or mod.padding == 0)
            outpad_ok = (mod.output_padding == (0,) or mod.output_padding == 0)
            groups_ok = (mod.groups == 1)
            use_triton = (
                x.is_cuda
                and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and stride_ok
                and padding_ok
                and outpad_ok
                and groups_ok
            )
        except Exception:
            use_triton = False

        if not use_triton:
            return mod(x)

        x_contig = x.contiguous()
        w = mod.weight.contiguous()  # [C_IN, C_OUT, K]
        b = mod.bias
        B, C_IN, L_IN = x_contig.shape
        _, C_OUT, K = w.shape

        # For stride=1, padding=0, output_padding=0: L_OUT = L_IN + K - 1
        L_OUT = L_IN + K - 1
        # Allocate output directly in input dtype to avoid an extra cast later
        y = torch.empty((B, C_OUT, L_OUT), device=x.device, dtype=x.dtype)

        # Launch kernel
        BLOCK_T = 128
        BLOCK_OC = 32   # keep >=16 for tl.dot; 32 improves throughput on small Cout tiles
        BLOCK_C = 64    # process more Cin per iter to reduce looping

        grid = (
            B,
            triton.cdiv(C_OUT, BLOCK_OC),
            triton.cdiv(L_OUT, BLOCK_T),
        )

        b_ptr = b.contiguous() if b is not None else y  # dummy if no bias
        _deconv1d_stride1_kernel[grid](
            x_contig,
            w,
            b_ptr,
            y,
            B,
            C_IN,
            C_OUT,
            L_IN,
            K,
            L_OUT,
            BLOCK_T=BLOCK_T,
            BLOCK_OC=BLOCK_OC,
            BLOCK_C=BLOCK_C,
            HAS_BIAS=(b is not None),
            num_warps=4,
            num_stages=3,
        )
        return y


# Test code
batch_size = 16
in_channels = 64
out_channels = 3
kernel_size = 3
length = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization