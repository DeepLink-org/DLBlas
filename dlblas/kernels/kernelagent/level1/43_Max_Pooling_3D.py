import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 16}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_W": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_W": 32}, num_warps=1, num_stages=4),
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=4),
    ],
    key=["outW"],
)
@triton.jit
def _maxpool3d_fwd_kernel(
    x_ptr,  # *f32 / *f16 contiguous NCDHW
    y_ptr,  # *f32 / *f16 contiguous NCDHW
    N, C, D, H, W,
    outD, outH, outW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_row = tl.program_id(0)  # over N*C*outD*outH
    pid_col = tl.program_id(1)  # over outW tiles

    ow = pid_col * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow < outW
    tl.max_contiguous(ow, BLOCK_W)
    tl.multiple_of(ow, 1)

    # decompose pid_row -> (n, c, od, oh)
    oh = pid_row % outH
    t = pid_row // outH
    od = t % outD
    t = t // outD
    c = t % C
    n = t // C

    # starting coords in input for this output location (vector across W)
    in_z0 = od * stride_d - pad_d
    in_y0 = oh * stride_h - pad_h
    in_x0 = ow * stride_w - pad_w  # [BLOCK_W]

    # flattened base offsets assuming contiguous memory
    # for x: ((((n*C + c) * D + z) * H + y) * W + x)
    base_nc = (n * C + c) * D * H * W

    # prepare output linear index (contiguous)
    out_idx = (((n * C + c) * outD + od) * outH + oh) * outW + ow

    # accumulator in fp32 for robustness (store will cast as needed)
    neg_inf = -float("inf")
    acc = tl.full((BLOCK_W,), neg_inf, dtype=tl.float32)

    # precompute level strides
    L1 = H * W
    L2 = W

    # Reorder loops to hoist x/x_valid computation out of kd/kh loops.
    x = in_x0
    for kw in tl.static_range(0, K_W):
        x_valid = (x >= 0) & (x < W)
        for kd in tl.static_range(0, K_D):
            z = in_z0 + kd * dil_d
            z_valid = (z >= 0) & (z < D)
            z_base = z * L1
            for kh in tl.static_range(0, K_H):
                y = in_y0 + kh * dil_h
                y_valid = (y >= 0) & (y < H)
                base_zh = z_base + y * L2
                m = mask_ow & z_valid & y_valid & x_valid
                in_idx = base_nc + base_zh + x
                vals = tl.load(x_ptr + in_idx, mask=m, other=neg_inf, eviction_policy="evict_first")
                vals = vals.to(tl.float32)
                acc = tl.maximum(acc, vals)
        x += dil_w

    tl.store(y_ptr + out_idx, acc, mask=mask_ow)


def _as_triple(v):
    if isinstance(v, (tuple, list)):
        assert len(v) == 3
        return int(v[0]), int(v[1]), int(v[2])
    v = int(v)
    return (v, v, v)


def _compute_out_dim(in_size: int, k: int, stride: int, pad: int, dil: int, ceil_mode: bool) -> int:
    # effective kernel size with dilation
    eff = dil * (k - 1) + 1
    if ceil_mode:
        return max(0, (in_size + 2 * pad - eff + stride) // stride)
    else:
        return max(0, (in_size + 2 * pad - eff) // stride + 1)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.
        """
        super(ModelNew, self).__init__()
        # Reference (fallback) module to guarantee semantics for unsupported cases
        self.maxpool_ref = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

        # Cache params for fast path
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.
        """
        # Triton fast path conditions
        use_triton = (
            x.is_cuda
            and x.is_contiguous()
            and not self.return_indices
            and x.dtype in (torch.float16, torch.float32)
        )
        if not use_triton:
            return self.maxpool_ref(x)

        N, C, D, H, W = x.shape

        kD, kH, kW = _as_triple(self.kernel_size)
        sD, sH, sW = _as_triple(self.stride)
        pD, pH, pW = _as_triple(self.padding)
        dD, dH, dW = _as_triple(self.dilation)

        outD = _compute_out_dim(D, kD, sD, pD, dD, self.ceil_mode)
        outH = _compute_out_dim(H, kH, sH, pH, dH, self.ceil_mode)
        outW = _compute_out_dim(W, kW, sW, pW, dW, self.ceil_mode)

        if outD == 0 or outH == 0 or outW == 0:
            return x.new_empty((N, C, outD, outH, outW))

        y = torch.empty((N, C, outD, outH, outW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N * C * outD * outH, triton.cdiv(outW, META["BLOCK_W"]))
        _maxpool3d_fwd_kernel[grid](
            x, y,
            N, C, D, H, W,
            outD, outH, outW,
            sD, sH, sW,
            pD, pH, pW,
            dD, dH, dW,
            K_D=kD, K_H=kH, K_W=kW,
        )
        return y


# Default problem setup (kept identical to the original)
batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]