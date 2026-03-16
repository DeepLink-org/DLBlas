import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_fwd_kernel(
    x_ptr, y_ptr,
    N, C, H, W, OH, OW,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_nc_oh = tl.program_id(0)  # over (N*C*OH)
    pid_w_blk = tl.program_id(1)  # over width tiles

    # derive n, c, oh
    nc = pid_nc_oh // OH
    oh = pid_nc_oh - nc * OH
    n = nc // C
    c = nc - n * C

    # offsets for output width
    offs_w = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    tl.multiple_of(offs_w, 16)
    tl.max_contiguous(offs_w, BLOCK_W)
    ow_mask = offs_w < OW

    # precompute base pointers
    x_base = x_ptr + n * in_stride_n + c * in_stride_c
    y_ptrs = y_ptr + n * out_stride_n + c * out_stride_c + oh * out_stride_h + offs_w * out_stride_w

    # top-left indices in input
    ih0 = oh * SH - PH                  # scalar
    iw0 = offs_w * SW - PW              # vector

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    if (PH == 0) and (PW == 0):
        # Fast path: no padding => all windows that produce outputs are fully in-bounds.
        # We still guard inactive lanes by ow_mask.
        row0 = x_base + ih0 * in_stride_h + iw0 * in_stride_w
        for kh in tl.static_range(0, KH):
            base_row = row0 + kh * in_stride_h
            for kw in tl.static_range(0, KW):
                vals = tl.load(base_row + kw * in_stride_w, mask=ow_mask, other=0.0)
                acc += vals  # implicit cast to fp32
        inv_kk = 1.0 / (KH * KW)
        out_vals = acc * inv_kk
    else:
        # General path: handle padding by computing per-lane valid element count
        # h_count is scalar (same for all lanes for given oh); w_count is per-lane
        h_lo = tl.maximum(0, -ih0)
        h_hi = tl.minimum(KH, H - ih0)
        h_cnt = tl.maximum(0, h_hi - h_lo)

        w_lo = tl.maximum(0, -iw0)
        w_hi = tl.minimum(KW, W - iw0)
        w_cnt = tl.maximum(0, w_hi - w_lo)

        denom_cnt = h_cnt * w_cnt
        denom_cnt = tl.where(ow_mask, denom_cnt, 1)
        denom_f = denom_cnt.to(tl.float32)

        for kh in tl.static_range(0, KH):
            ih = ih0 + kh
            mask_h = (ih >= 0) & (ih < H)
            row_ptr = x_base + ih * in_stride_h
            base_ptrs = row_ptr + iw0 * in_stride_w
            for kw in tl.static_range(0, KW):
                iw = iw0 + kw
                mask = ow_mask & mask_h & (iw >= 0) & (iw < W)
                vals = tl.load(base_ptrs + kw * in_stride_w, mask=mask, other=0.0)
                acc += vals  # implicit cast to fp32

        out_vals = acc / denom_f

    tl.store(y_ptrs, out_vals, mask=ow_mask)


@triton.jit
def avg_pool2d_fwd_kernel_3x3(
    x_ptr, y_ptr,
    N, C, H, W, OH, OW,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    KH: tl.constexpr, KW: tl.constexpr,  # kept for signature compatibility
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_nc_oh = tl.program_id(0)
    pid_w_blk = tl.program_id(1)

    # decode n, c, oh
    nc = pid_nc_oh // OH
    oh = pid_nc_oh - nc * OH
    n = nc // C
    c = nc - n * C

    offs_w = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    tl.multiple_of(offs_w, 16)
    tl.max_contiguous(offs_w, BLOCK_W)
    ow_mask = offs_w < OW

    x_base = x_ptr + n * in_stride_n + c * in_stride_c
    y_ptrs = y_ptr + n * out_stride_n + c * out_stride_c + oh * out_stride_h + offs_w * out_stride_w

    ih0 = oh * SH - PH
    iw0 = offs_w * SW - PW

    if (PH == 0) and (PW == 0):
        # Fast 3x3 path without padding: 9 loads guarded only by ow_mask
        base0 = x_base + ih0 * in_stride_h + iw0 * in_stride_w
        base1 = base0 + in_stride_h
        base2 = base1 + in_stride_h

        r00 = tl.load(base0 + 0 * in_stride_w, mask=ow_mask, other=0.0)
        r01 = tl.load(base0 + 1 * in_stride_w, mask=ow_mask, other=0.0)
        r02 = tl.load(base0 + 2 * in_stride_w, mask=ow_mask, other=0.0)

        r10 = tl.load(base1 + 0 * in_stride_w, mask=ow_mask, other=0.0)
        r11 = tl.load(base1 + 1 * in_stride_w, mask=ow_mask, other=0.0)
        r12 = tl.load(base1 + 2 * in_stride_w, mask=ow_mask, other=0.0)

        r20 = tl.load(base2 + 0 * in_stride_w, mask=ow_mask, other=0.0)
        r21 = tl.load(base2 + 1 * in_stride_w, mask=ow_mask, other=0.0)
        r22 = tl.load(base2 + 2 * in_stride_w, mask=ow_mask, other=0.0)

        # accumulate in fp32 with implicit upcast
        s0 = (r00 + r01) + r02
        s1 = (r10 + r11) + r12
        s2 = (r20 + r21) + r22
        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        acc += s0
        acc += s1
        acc += s2

        out_vals = acc * (1.0 / 9.0)
    else:
        # General 3x3 with padding: compute masks per tap and count valid
        iy0 = ih0 + 0
        iy1 = ih0 + 1
        iy2 = ih0 + 2

        vy0 = (iy0 >= 0) & (iy0 < H)
        vy1 = (iy1 >= 0) & (iy1 < H)
        vy2 = (iy2 >= 0) & (iy2 < H)

        vx0 = (iw0 + 0 >= 0) & (iw0 + 0 < W)
        vx1 = (iw0 + 1 >= 0) & (iw0 + 1 < W)
        vx2 = (iw0 + 2 >= 0) & (iw0 + 2 < W)

        base0 = x_base + iy0 * in_stride_h + iw0 * in_stride_w
        base1 = x_base + iy1 * in_stride_h + iw0 * in_stride_w
        base2 = x_base + iy2 * in_stride_h + iw0 * in_stride_w

        m00 = ow_mask & vy0 & vx0
        m01 = ow_mask & vy0 & vx1
        m02 = ow_mask & vy0 & vx2

        m10 = ow_mask & vy1 & vx0
        m11 = ow_mask & vy1 & vx1
        m12 = ow_mask & vy1 & vx2

        m20 = ow_mask & vy2 & vx0
        m21 = ow_mask & vy2 & vx1
        m22 = ow_mask & vy2 & vx2

        r00 = tl.load(base0 + 0 * in_stride_w, mask=m00, other=0.0)
        r01 = tl.load(base0 + 1 * in_stride_w, mask=m01, other=0.0)
        r02 = tl.load(base0 + 2 * in_stride_w, mask=m02, other=0.0)

        r10 = tl.load(base1 + 0 * in_stride_w, mask=m10, other=0.0)
        r11 = tl.load(base1 + 1 * in_stride_w, mask=m11, other=0.0)
        r12 = tl.load(base1 + 2 * in_stride_w, mask=m12, other=0.0)

        r20 = tl.load(base2 + 0 * in_stride_w, mask=m20, other=0.0)
        r21 = tl.load(base2 + 1 * in_stride_w, mask=m21, other=0.0)
        r22 = tl.load(base2 + 2 * in_stride_w, mask=m22, other=0.0)

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        acc += r00; acc += r01; acc += r02
        acc += r10; acc += r11; acc += r12
        acc += r20; acc += r21; acc += r22

        cnt = (
            tl.where(m00, 1.0, 0.0) + tl.where(m01, 1.0, 0.0) + tl.where(m02, 1.0, 0.0) +
            tl.where(m10, 1.0, 0.0) + tl.where(m11, 1.0, 0.0) + tl.where(m12, 1.0, 0.0) +
            tl.where(m20, 1.0, 0.0) + tl.where(m21, 1.0, 0.0) + tl.where(m22, 1.0, 0.0)
        ).to(tl.float32)
        cnt = tl.maximum(cnt, 1.0)
        out_vals = acc / cnt

    tl.store(y_ptrs, out_vals, mask=ow_mask)


def _avg_pool2d_triton(x: torch.Tensor, kernel_size: int, stride: int | None = None, padding: int = 0):
    # Fallback to PyTorch if not CUDA or Triton unavailable
    if (not x.is_cuda) or (x.numel() == 0):
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    if stride is None:
        stride = kernel_size

    # shapes
    N, C, H, W = x.shape
    KH = KW = int(kernel_size)
    SH = SW = int(stride)
    PH = PW = int(padding)

    # output sizes (floor as in PyTorch AvgPool2d)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    assert OH > 0 and OW > 0, "Invalid output size; check kernel/stride/padding."

    # allocate output
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # get strides in elements
    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()

    # choose tile size
    USE_3X3 = (KH == 3) and (KW == 3)
    BLOCK_W = 128

    # grid: one program per (n, c, oh) row, and tiles along width
    grid = (N * C * OH, triton.cdiv(OW, BLOCK_W))

    # launch
    if USE_3X3:
        avg_pool2d_fwd_kernel_3x3[grid](
            x, y,
            N, C, H, W, OH, OW,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            out_stride_n, out_stride_c, out_stride_h, out_stride_w,
            KH=KH, KW=KW, SH=SH, SW=SW, PH=PH, PW=PW,
            BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )
    else:
        avg_pool2d_fwd_kernel[grid](
            x, y,
            N, C, H, W, OH, OW,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            out_stride_n, out_stride_c, out_stride_h, out_stride_w,
            KH=KH, KW=KW, SH=SH, SW=SW, PH=PH, PW=PW,
            BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2
        )
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling using a custom Triton kernel on CUDA.
    Falls back to PyTorch implementation on CPU.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else None
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel on CUDA, fallback to PyTorch otherwise
        return _avg_pool2d_triton(x, self.kernel_size, self.stride, self.padding)


# default problem size for benchmarking
batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size]