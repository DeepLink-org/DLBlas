import torch
import torch.nn as nn
import triton
import triton.language as tl


def _triple(v):
    if isinstance(v, tuple):
        assert len(v) == 3
        return v
    return (v, v, v)


@triton.jit
def avgpool3d_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    SD, SH, SW,
    PD, PH, PW,
    n_elements,
    KSIZE_D: tl.constexpr, KSIZE_H: tl.constexpr, KSIZE_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_o = offs < n_elements

    # Unravel linear index into (n, c, od, oh, ow)
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    t = t // OD
    c = t % C
    n = t // C

    # Input start indices for each output element
    id_base = od * SD - PD
    ih_base = oh * SH - PH
    iw_base = ow * SW - PW

    # Accumulator in float32
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Flattened base for nc
    base_nc = n * C + c
    base_ncD = base_nc * D

    # Iterate kernel window using unrolled static ranges
    for kd in tl.static_range(KSIZE_D):
        idv = id_base + kd
        md = (idv >= 0) & (idv < D)
        for kh in tl.static_range(KSIZE_H):
            ihv = ih_base + kh
            mh = (ihv >= 0) & (ihv < H)

            m_dh = mask_o & md & mh

            # Compute row base pointer once per (kd, kh)
            row_base = ((base_ncD + idv) * H + ihv) * W
            ptr_row = x_ptr + row_base

            # Start pointer for kw loop; increment by +1 each iteration (contiguous in W)
            p = ptr_row + iw_base

            # Generic unrolled kw loop
            for kw in tl.static_range(KSIZE_W):
                iwv = iw_base + kw
                mw = (iwv >= 0) & (iwv < W)
                m = m_dh & mw
                v = tl.load(p, mask=m, other=0.0)
                acc += v.to(tl.float32)
                p += 1

    # Average: divide by full kernel volume (count_include_pad=True)
    scale = 1.0 / float(KSIZE_D * KSIZE_H * KSIZE_W)
    out = acc * scale

    tl.store(y_ptr + offs, out, mask=mask_o)


class ModelNew(nn.Module):
    """
    Simple model that performs 3D Average Pooling using a Triton kernel on CUDA.
    Falls back to PyTorch AvgPool3d on CPU.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if not CUDA
        if not x.is_cuda:
            return self.avg_pool(x)

        x = x.contiguous()
        N, C, D, H, W = x.shape

        kD, kH, kW = _triple(self.avg_pool.kernel_size)
        sD, sH, sW = _triple(self.avg_pool.stride if self.avg_pool.stride is not None else self.avg_pool.kernel_size)
        pD, pH, pW = _triple(self.avg_pool.padding)

        # Output dimensions (ceil_mode=False)
        OD = (D + 2 * pD - kD) // sD + 1
        OH = (H + 2 * pH - kH) // sH + 1
        OW = (W + 2 * pW - kW) // sW + 1

        # Allocate output with same dtype as input
        y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

        n_elements = y.numel()
        # Tuned BLOCK for good occupancy and memory throughput on H200
        BLOCK = 256
        grid = lambda META: (triton.cdiv(n_elements, META['BLOCK']),)

        avgpool3d_kernel[grid](
            x, y,
            N, C, D, H, W,
            OD, OH, OW,
            sD, sH, sW,
            pD, pH, pW,
            n_elements,
            KSIZE_D=kD, KSIZE_H=kH, KSIZE_W=kW,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=4,
        )
        return y


batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    # Run on CUDA for benchmarking the Triton kernel path
    x = torch.randn(batch_size, channels, depth, height, width, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]