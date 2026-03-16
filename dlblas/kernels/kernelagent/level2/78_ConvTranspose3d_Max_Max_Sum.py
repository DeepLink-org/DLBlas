import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 128}, num_warps=8, num_stages=2),
    ],
    key=['H2', 'W2'],
)
@triton.jit
def _maxpool_6x_3d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    D2, H2, W2,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ndc = tl.program_id(2)

    # Decode n, c, d_out from pid_ndc
    CD2 = C * D2
    n = pid_ndc // CD2
    rem = pid_ndc % CD2
    c = rem // D2
    d_out = rem % D2

    # Tile of output H/W
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mw = w_out < W2
    mh = h_out < H2
    m_hw = mh[:, None] & mw[None, :]

    # Compute starting indices in input for this output tile
    d_start = d_out * 6
    h_start = h_out * 6  # [BH]
    w_start = w_out * 6  # [BW]

    # Prepare max accumulator in fp32 for numerical stability
    m = tl.full((BLOCK_H, BLOCK_W), -float('inf'), dtype=tl.float32)

    # Base pointer for batch/channel
    base_nc = n * stride_n + c * stride_c

    # Iterate over the 6x6x6 window
    for kd in range(6):
        d_idx_off = (d_start + kd) * stride_d
        for kh in range(6):
            h_idx = h_start[:, None] + kh
            h_off = h_idx * stride_h
            for kw in range(6):
                w_idx = w_start[None, :] + kw
                w_off = w_idx * stride_w
                ptr = x_ptr + base_nc + d_idx_off + h_off + w_off
                val = tl.load(ptr, mask=m_hw, other=0.0)
                m = tl.where(m_hw, tl.maximum(m, val.to(tl.float32)), m)

    # Store results
    out_base = out_ptr + n * out_stride_n + c * out_stride_c + d_out * out_stride_d
    out_ptrs = out_base + h_out[:, None] * out_stride_h + w_out[None, :] * out_stride_w
    tl.store(out_ptrs, m, mask=m_hw)


def _fused_two_pools_into_one(x: torch.Tensor) -> torch.Tensor:
    """
    Replace consecutive MaxPool3d(kernel=2, stride=2) and MaxPool3d(kernel=3, stride=3)
    with a single 3D max-pool using kernel=6, stride=6 (equivalent composition).
    Implemented as a Triton kernel that keeps per-channel outputs.
    """
    # Input dims: N, C, D, H, W
    N, C, D, H, W = x.shape
    # Output dims after two pools: floor((D-6)/6)+1 etc. (equivalent to composing 2 and 3)
    D2 = (D - 6) // 6 + 1
    H2 = (H - 6) // 6 + 1
    W2 = (W - 6) // 6 + 1

    # Allocate output tensor (per-channel pooled)
    out = torch.empty((N, C, D2, H2, W2), device=x.device, dtype=torch.float32)

    # Strides in elements (PyTorch gives strides in elements already)
    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oD, oH, oW = out.stride()

    # Grid: (W tiles, H tiles, N*C*D2)
    def grid(meta):
        return (
            triton.cdiv(W2, meta['BLOCK_W']),
            triton.cdiv(H2, meta['BLOCK_H']),
            N * C * D2,
        )

    _maxpool_6x_3d_kernel[grid](
        x, out,
        N, C, D, H, W,
        sN, sC, sD, sH, sW,
        oN, oC, oD, oH, oW,
        D2, H2, W2,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Keep original modules for correctness and CPU fallback
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use Triton fused pooling when on CUDA; else fall back to PyTorch sequence
        if x.is_cuda:
            # Triton kernel expects fp32 for best numerical stability
            if x.dtype != torch.float32:
                x_fp32 = x.float()
            else:
                x_fp32 = x
            pooled = _fused_two_pools_into_one(x_fp32)
            x = pooled.sum(dim=1, keepdim=True)
            # Match original dtype if it wasn't fp32
            if x_fp32 is not x:
                x = x.to(x_fp32.dtype)
        else:
            x = self.max_pool1(x)
            x = self.max_pool2(x)
            x = torch.sum(x, dim=1, keepdim=True)
        return x


batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]