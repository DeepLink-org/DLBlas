import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _copy_into_channel_slice_kernel(
    src_ptr,  # *T
    dst_ptr,  # *T
    total_elems,  # int: (unused, kept for signature compatibility)
    chunk_elems,  # int: elements per batch (C*H*W of src)
    src_batch_stride,  # int: stride between batches for src (in elements)
    dst_batch_stride,  # int: stride between batches for dst (in elements)
    dst_c_offset_elems,  # int: starting offset in dst (in elements)
    BLOCK: tl.constexpr,
):
    # 2D launch: pid_n over batches, pid_blk over contiguous blocks within a batch
    pid_n = tl.program_id(0)
    pid_blk = tl.program_id(1)

    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < chunk_elems

    # Help compiler vectorize and coalesce
    tl.max_contiguous(offs, BLOCK)
    tl.multiple_of(offs, 16)

    src_ix = pid_n * src_batch_stride + offs
    dst_ix = pid_n * dst_batch_stride + dst_c_offset_elems + offs

    val = tl.load(src_ptr + src_ix, mask=mask, other=0, cache_modifier=".cg")
    tl.store(dst_ptr + dst_ix, val, mask=mask)


@triton.jit
def _copy_into_channel_slice_channels_last_kernel(
    src_ptr,  # *T
    dst_ptr,  # *T
    NHW,      # int: N*H*W
    C_LOCAL,  # int: channels in src
    C_TOTAL,  # int: total channels in dst
    C_OFFSET, # int: starting channel offset in dst
    BLOCK_NHW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 2D tiling over NHW and C_LOCAL for channels-last contiguous tensors
    pid_nhw = tl.program_id(0)
    pid_c = tl.program_id(1)

    nhw_offsets = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_nhw = nhw_offsets < NHW
    mask_c = c_offsets < C_LOCAL
    mask = mask_nhw[:, None] & mask_c[None, :]

    # Hints to help vectorization along C and NHW
    tl.max_contiguous(c_offsets, BLOCK_C)
    tl.max_contiguous(nhw_offsets, BLOCK_NHW)
    tl.multiple_of(c_offsets, 16)

    # For channels-last: each (n,h,w) has contiguous C dimension
    src_ix = nhw_offsets[:, None] * C_LOCAL + c_offsets[None, :]
    dst_ix = nhw_offsets[:, None] * C_TOTAL + (C_OFFSET + c_offsets[None, :])

    vals = tl.load(src_ptr + src_ix, mask=mask, other=0, cache_modifier=".cg")
    tl.store(dst_ptr + dst_ix, vals, mask=mask)


def _copy_into_channel_slice_channels_last(dst: torch.Tensor, src: torch.Tensor, c_offset: int):
    """
    Fast copy for channels_last-contiguous tensors:
      src (N,C,H,W) -> dst (N,C_total,H,W) at channel offset c_offset.
    """
    assert src.is_cuda and dst.is_cuda, "Requires CUDA tensors"
    assert src.is_contiguous(memory_format=torch.channels_last), "src must be channels_last contiguous"
    assert dst.is_contiguous(memory_format=torch.channels_last), "dst must be channels_last contiguous"
    N, C, H, W = src.shape
    NHW = N * H * W
    C_total = dst.shape[1]
    assert 0 <= c_offset <= (C_total - C)
    if NHW == 0 or C == 0:
        return

    # Tile sizes tuned for Hopper/H200; growth_rate=32 benefits from BLOCK_C=32
    BLOCK_C = 32
    BLOCK_NHW = 512
    grid = (triton.cdiv(NHW, BLOCK_NHW), triton.cdiv(C, BLOCK_C))
    _copy_into_channel_slice_channels_last_kernel[grid](
        src, dst, NHW, C, C_total, c_offset,
        BLOCK_NHW=BLOCK_NHW,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=2,
    )


def _copy_into_channel_slice(dst: torch.Tensor, src: torch.Tensor, c_offset: int):
    """
    Copy src (N, C, H, W) into dst (N, C_total, H, W) starting at channel offset c_offset.
    Assumes both tensors are contiguous in NCHW layout.
    """
    assert src.is_contiguous(), "src must be contiguous"
    assert dst.is_contiguous(), "dst must be contiguous"
    assert src.device.type == dst.device.type == "cuda", "Kernel requires CUDA tensors"
    N, C, H, W = src.shape
    total_elems = N * C * H * W
    if total_elems == 0:
        return
    chunk_elems = C * H * W  # per-batch contiguous block
    src_batch_stride = src.stride(0)
    dst_batch_stride = dst.stride(0)
    dst_c_offset_elems = c_offset * H * W

    # Use a 2D grid: one dimension over batches, one over blocks in a batch
    BLOCK = 8192
    grid = (N, triton.cdiv(chunk_elems, BLOCK))
    _copy_into_channel_slice_kernel[grid](
        src,
        dst,
        total_elems,  # kept for signature compatibility
        chunk_elems,
        src_batch_stride,
        dst_batch_stride,
        dst_c_offset_elems,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=2,
    )


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        # Enable cuDNN autotuning for faster convs on fixed shapes
        if x.is_cuda and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        B, Cin, H, W = x.shape
        assert Cin == self.num_input_features, "Input channel size mismatch"
        Cout_total = Cin + len(self.layers) * self.growth_rate

        # Prefer channels_last on CUDA to speed up convolutions
        if x.is_cuda:
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            out_buf = torch.empty((B, Cout_total, H, W), device=x.device, dtype=x.dtype,
                                  memory_format=torch.channels_last)
            # Copy initial input into the buffer at offset 0
            if x.is_contiguous(memory_format=torch.channels_last) and out_buf.is_contiguous(memory_format=torch.channels_last):
                _copy_into_channel_slice_channels_last(out_buf, x, 0)
            else:
                out_buf[:, :Cin].copy_(x)
        else:
            out_buf = torch.empty((B, Cout_total, H, W), device=x.device, dtype=x.dtype)
            out_buf[:, :Cin].copy_(x)

        # Current view corresponds to concatenation so far
        x_view = out_buf[:, :Cin]

        # Iteratively compute new features and place them into the buffer
        for i, layer in enumerate(self.layers):
            new_feature = layer(x_view)
            c0 = Cin + i * self.growth_rate
            c1 = c0 + self.growth_rate
            if new_feature.is_cuda and out_buf.is_contiguous(memory_format=torch.channels_last) and new_feature.is_contiguous(memory_format=torch.channels_last):
                _copy_into_channel_slice_channels_last(out_buf, new_feature, c0)
            elif new_feature.is_cuda and new_feature.is_contiguous() and out_buf.is_contiguous():
                _copy_into_channel_slice(out_buf, new_feature, c0)
            else:
                out_buf[:, c0:c1].copy_(new_feature)
            # Update the view to include the newly added features (equivalent to torch.cat)
            x_view = out_buf[:, :c1]

        return x_view


batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224


def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]


def get_init_inputs():
    return [num_layers, num_input_features, growth_rate]