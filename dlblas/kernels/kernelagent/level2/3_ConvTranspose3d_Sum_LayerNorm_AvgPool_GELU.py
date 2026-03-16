import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    TRITON_AVAILABLE = True
    HAS_LIBDEVICE = True
except Exception:
    TRITON_AVAILABLE = False
    HAS_LIBDEVICE = False


@triton.jit
def _add_layernorm_lastdim_kernel(
    x_ptr,         # *flattened* input pointer (contiguous)
    y_ptr,         # *flattened* output pointer (contiguous)
    gamma_ptr,     # weight (normalized_shape)
    beta_ptr,      # bias   (normalized_shape)
    sum_w,         # scalar to add before LN (invariance: LN(x + c) == LN(x))
    M,             # number of columns (normalized dimension)
    N_ROWS,        # number of rows (total elements // M)
    eps,           # epsilon
    BLOCK_SIZE: tl.constexpr,   # >= M
    ROWS_PER_CTA: tl.constexpr  # rows processed per CTA
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(cols, 16)
    tl.max_contiguous(cols, BLOCK_SIZE)

    # Reuse gamma/beta across rows (kept in registers)
    col_mask = cols < M
    g = tl.load(gamma_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    inv_M = 1.0 / tl.full((), M, dtype=tl.float32)

    # Base row index for this program id
    row_base = pid * ROWS_PER_CTA

    # Prefetch first row
    row_idx0 = row_base
    row_active0 = row_idx0 < N_ROWS
    row_start0 = row_idx0 * M
    mask0 = col_mask & row_active0
    x_raw0 = tl.load(x_ptr + row_start0 + cols, mask=mask0, other=0.0)

    # Software-pipelined loop: prefetch next row while computing current
    for r in tl.static_range(ROWS_PER_CTA):
        row_idx = row_base + r
        row_active = row_idx < N_ROWS
        mask = col_mask & row_active

        # Current row data; exploit LN invariance to skip adding sum_w
        x_fp = x_raw0.to(tl.float32)

        # Prefetch next row early to hide latency
        if r < ROWS_PER_CTA - 1:
            next_idx = row_idx + 1
            next_active = next_idx < N_ROWS
            next_start = next_idx * M
            next_mask = col_mask & next_active
            x_raw1 = tl.load(x_ptr + next_start + cols, mask=next_mask, other=0.0)

        # Mean/Var in fp32
        mean = tl.sum(x_fp, axis=0) * inv_M
        diff = x_fp - mean
        var = tl.sum(diff * diff, axis=0) * inv_M
        rstd = tl.rsqrt(var + eps)

        y_fp = (diff * rstd) * g + b
        y = y_fp.to(x_raw0.dtype)
        tl.store(y_ptr + row_idx * M + cols, y, mask=mask)

        if r < ROWS_PER_CTA - 1:
            x_raw0 = x_raw1


@triton.jit
def _avgpool3d_gelu_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    TOT_ROWS,
    BLOCK_W: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr
):
    pid = tl.program_id(axis=0)
    w = tl.arange(0, BLOCK_W)

    # Precompute input/output strides
    in_stride_w = 1
    in_stride_h = W
    in_stride_d = H * W
    in_stride_c = D * H * W
    in_stride_n = C * D * H * W

    out_stride_w = 1
    out_stride_h = Wo
    out_stride_d = Ho * Wo
    out_stride_c = Do * Ho * Wo
    out_stride_n = C * Do * Ho * Wo

    for r in tl.static_range(ROWS_PER_CTA):
        row = pid * ROWS_PER_CTA + r
        row_mask = row < TOT_ROWS

        # Decode (n, c, do, ho) from row id
        ho = row % Ho
        tmp = row // Ho
        do = tmp % Do
        tmp = tmp // Do
        c = tmp % C
        n = tmp // C

        # Base indices
        di0 = do * KD
        hi0 = ho * KH

        base_in = n * in_stride_n + c * in_stride_c + di0 * in_stride_d + hi0 * in_stride_h
        w_mask = (w < Wo) & row_mask

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        # Accumulate over kernel window
        for kd_i in tl.static_range(KD):
            base_kd = base_in + kd_i * in_stride_d
            for kh_i in tl.static_range(KH):
                base_kh = base_kd + kh_i * in_stride_h
                for kw_i in tl.static_range(KW):
                    offs = base_kh + w * KW + kw_i
                    val = tl.load(x_ptr + offs, mask=w_mask, other=0.0)
                    acc += val.to(tl.float32)

        # Average
        scale = 1.0 / (KD * KH * KW)
        avg = acc * scale

        # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        e = libdevice.erf(avg * 0.7071067811865476)  # 1/sqrt(2)
        out = 0.5 * avg * (1.0 + e)

        base_out = n * out_stride_n + c * out_stride_c + do * out_stride_d + ho * out_stride_h
        tl.store(y_ptr + base_out + w, out, mask=w_mask)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def fused_add_layernorm_lastdim(x: torch.Tensor, sum_weight: torch.Tensor, ln_mod: nn.LayerNorm) -> torch.Tensor:
    # Preconditions: normalize across last dim
    M = x.shape[-1]
    assert isinstance(ln_mod.normalized_shape, (tuple, list)) and len(ln_mod.normalized_shape) == 1, \
        "This fused kernel only supports normalization over the last single dimension."
    assert ln_mod.normalized_shape[0] == M, "normalized_shape must match the last dimension size"

    # Ensure contiguous memory layout
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    # Get gamma/beta (elementwise_affine may be False)
    if getattr(ln_mod, "elementwise_affine", True):
        gamma = ln_mod.weight
        beta = ln_mod.bias
    else:
        gamma = torch.ones(M, device=x.device, dtype=x.dtype)
        beta = torch.zeros(M, device=x.device, dtype=x.dtype)

    eps = ln_mod.eps if hasattr(ln_mod, "eps") else 1e-5

    # Flatten to [N_ROWS, M]
    N_ROWS = x_contig.numel() // M

    # Kernel launch params
    BLOCK_SIZE = _next_power_of_2(M)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    # Process more rows per CTA to reduce launch overhead and improve data reuse
    ROWS_PER_CTA = 32

    grid = (triton.cdiv(N_ROWS, ROWS_PER_CTA),)

    # Heuristic for warps: keep conservative to avoid register pressure for small M
    if BLOCK_SIZE <= 64:
        num_warps = 1
    elif BLOCK_SIZE <= 128:
        num_warps = 2
    elif BLOCK_SIZE <= 256:
        num_warps = 4
    else:
        num_warps = 8

    _add_layernorm_lastdim_kernel[grid](
        x_contig.view(-1), y.view(-1),
        gamma, beta,
        float(sum_weight.item()),
        M, N_ROWS, eps,
        BLOCK_SIZE=BLOCK_SIZE, ROWS_PER_CTA=ROWS_PER_CTA,
        num_warps=num_warps, num_stages=4
    )
    return y


def _to_3tuple(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 3:
            return tuple(int(x) for x in v)
        elif len(v) == 1:
            return (int(v[0]), int(v[0]), int(v[0]))
        else:
            return (int(v[0]), int(v[0]), int(v[0]))
    else:
        v = int(v)
        return (v, v, v)


def fused_avgpool3d_gelu(x: torch.Tensor, avg_mod: nn.AvgPool3d) -> torch.Tensor | None:
    # Only supports common/default AvgPool3d: stride == kernel, padding == 0, ceil_mode == False,
    # count_include_pad == True, divisor_override is None
    kd, kh, kw = _to_3tuple(avg_mod.kernel_size)
    stride = avg_mod.stride
    if stride is None:
        stride = (kd, kh, kw)
    else:
        stride = _to_3tuple(stride)
    padding = _to_3tuple(avg_mod.padding)
    if not (stride == (kd, kh, kw) and padding == (0, 0, 0) and
            getattr(avg_mod, "ceil_mode", False) is False and
            getattr(avg_mod, "count_include_pad", True) is True and
            getattr(avg_mod, "divisor_override", None) is None):
        return None

    N, C, D, H, W = x.shape
    # Output dims (ceil_mode=False)
    Do = (D - kd) // kd + 1
    Ho = (H - kh) // kh + 1
    Wo = (W - kw) // kw + 1

    x_contig = x.contiguous()
    y = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_W = _next_power_of_2(Wo)
    BLOCK_W = min(BLOCK_W, 1024)
    ROWS_PER_CTA = 8

    TOT_ROWS = N * C * Do * Ho
    grid = (triton.cdiv(TOT_ROWS, ROWS_PER_CTA),)

    # Warps heuristic
    if BLOCK_W <= 64:
        num_warps = 2
    elif BLOCK_W <= 128:
        num_warps = 4
    else:
        num_warps = 8

    _avgpool3d_gelu_kernel[grid](
        x_contig, y,
        N, C, D, H, W,
        Do, Ho, Wo,
        TOT_ROWS,
        BLOCK_W=BLOCK_W, ROWS_PER_CTA=ROWS_PER_CTA,
        KD=kd, KH=kh, KW=kw,
        num_warps=num_warps, num_stages=2
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused: scalar add + layernorm over last dimension
        can_fuse = (
            TRITON_AVAILABLE and
            x.is_cuda and
            isinstance(self.norm.normalized_shape, (tuple, list)) and
            len(self.norm.normalized_shape) == 1 and
            self.norm.normalized_shape[0] == x.shape[-1]
        )
        if can_fuse:
            x = fused_add_layernorm_lastdim(x, self.sum_weight, self.norm)
        else:
            x = x + self.sum_weight
            x = self.norm(x)

        # Fused AvgPool3d + GELU when configuration is compatible
        can_fuse_pool = TRITON_AVAILABLE and HAS_LIBDEVICE and x.is_cuda
        y = None
        if can_fuse_pool:
            y = fused_avgpool3d_gelu(x, self.avg_pool)
        if y is None:
            x = self.avg_pool(x)
            x = self.gelu(x)
        else:
            x = y
        return x

batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]