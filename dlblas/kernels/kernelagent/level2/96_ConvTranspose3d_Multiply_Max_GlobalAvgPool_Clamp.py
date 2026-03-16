import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_scale_maxpool3d_gap_clamp(
    x_ptr,                       # *x_dtype [N, C, D, H, W] contiguous
    out_ptr,                     # *x_dtype [N*C] flattened output
    N, C, D, H, W,               # dimensions
    scale,                       # scalar
    clamp_min, clamp_max,        # scalars
    KSIZE: tl.constexpr,         # pooling kernel size (assumed stride=KSIZE, padding=0, dilation=1)
    DP: tl.constexpr,            # pooled D
    HP: tl.constexpr,            # pooled H
    WP: tl.constexpr,            # pooled W
    NWINS: tl.constexpr,         # total pooling windows = DP * HP * WP
    BLOCK_WINS: tl.constexpr,    # number of windows processed per iteration
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    # Strides for a contiguous [N, C, D, H, W]
    sN = C * D * H * W
    sC = D * H * W
    sD = H * W
    sH = W
    sW = 1

    base = x_ptr + n * sN + c * sC

    offs = tl.arange(0, BLOCK_WINS)
    sumv = tl.zeros((), dtype=tl.float32)

    hpwp = HP * WP

    for start in range(0, NWINS, BLOCK_WINS):
        idx = start + offs
        mask = idx < NWINS

        dp = idx // hpwp
        rem = idx - dp * hpwp
        hp = rem // WP
        wp = rem - hp * WP

        di0 = dp * KSIZE
        hi0 = hp * KSIZE
        wi0 = wp * KSIZE

        base_offsets = di0 * sD + hi0 * sH + wi0 * sW

        # Initialize maxima as -inf in fp32
        maxi = tl.full(offs.shape, -float("inf"), tl.float32)

        # Iterate KSIZE^3 inside the window
        for kd in tl.static_range(0, KSIZE):
            for kh in tl.static_range(0, KSIZE):
                for kw in tl.static_range(0, KSIZE):
                    ptrs = base + base_offsets + kd * sD + kh * sH + kw * sW
                    v = tl.load(ptrs, mask=mask, other=-float("inf"))
                    vf32 = v.to(tl.float32) * scale
                    # For masked lanes, set to -inf so they don't affect max
                    vf32 = tl.where(mask, vf32, -float("inf"))
                    maxi = tl.maximum(maxi, vf32)

        # Sum valid maxima of this chunk
        maxi = tl.where(mask, maxi, 0.0)
        part = tl.sum(maxi, axis=0, dtype=tl.float32)
        sumv += part

    denom = tl.full((), NWINS, tl.float32)
    meanv = sumv / denom
    meanv = tl.minimum(tl.maximum(meanv, clamp_min), clamp_max)

    out_off = n * C + c
    tl.store(out_ptr + out_off, meanv)


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        # Keep ConvTranspose3d in cuDNN
        y = self.conv_transpose(x)

        # Triton fast path: fuse scale -> maxpool3d (k=stride, p=0, d=1, ceil_mode=False) -> GAP -> clamp
        can_fuse = (
            y.is_cuda
            and y.is_contiguous()
        )
        if can_fuse:
            # Validate that pooling config matches our fused kernel assumptions
            k = self.maxpool.kernel_size
            s = getattr(self.maxpool, "stride", None)
            p = getattr(self.maxpool, "padding", 0)
            dila = getattr(self.maxpool, "dilation", 1)
            ceil_m = getattr(self.maxpool, "ceil_mode", False)

            if isinstance(k, (tuple, list)):
                if not (k[0] == k[1] == k[2]):
                    can_fuse = False
                else:
                    k = k[0]

            if s is None:
                s = k
            elif isinstance(s, (tuple, list)):
                if not (s[0] == s[1] == s[2] == k):
                    can_fuse = False
                else:
                    s = s[0]

            if isinstance(p, (tuple, list)):
                can_fuse = can_fuse and (p[0] == p[1] == p[2] == 0)
            else:
                can_fuse = can_fuse and (p == 0)

            if isinstance(dila, (tuple, list)):
                can_fuse = can_fuse and (dila[0] == dila[1] == dila[2] == 1)
            else:
                can_fuse = can_fuse and (dila == 1)

            can_fuse = can_fuse and (s == k) and (ceil_m is False)

        if can_fuse:
            N, C, D, H, W = y.shape
            # Compute output pooled sizes (floor)
            if D < k or H < k or W < k:
                can_fuse = False

        if can_fuse:
            DP = (D - k) // k + 1
            HP = (H - k) // k + 1
            WP = (W - k) // k + 1
            NWINS = DP * HP * WP

            # Allocate output [N, C, 1, 1, 1]
            out = torch.empty((N, C, 1, 1, 1), device=y.device, dtype=y.dtype)

            grid = (N * C,)
            _fused_scale_maxpool3d_gap_clamp[grid](
                y, out.view(-1),
                N, C, D, H, W,
                float(self.scale),
                float(self.clamp_min), float(self.clamp_max),
                KSIZE=k,
                DP=DP, HP=HP, WP=WP, NWINS=NWINS,
                BLOCK_WINS=256,
                num_warps=8,
                num_stages=2,
            )
            return out

        # Fallback path (original semantics)
        y = y * self.scale
        y = self.maxpool(y)
        y = self.global_avg_pool(y)
        y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
        return y


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]