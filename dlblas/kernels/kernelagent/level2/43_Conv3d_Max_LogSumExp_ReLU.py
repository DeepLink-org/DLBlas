import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def _pool_lse_relu_stream_kernel(
    x_ptr,           # *input  (N, C, D, H, W)
    y_ptr,           # *output (N, 1, DO, HO, WO)
    M,               # total positions over (N, DO, HO, WO)
    DO, HO, WO,      # pooled spatial sizes
    sxn, sxc, sxd, sxh, sxw,  # input strides
    syn, syc, syd, syh, syw,  # output strides
    C: tl.constexpr,          # channels to reduce over (compile-time)
    BLOCK: tl.constexpr,      # vector width
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    # Decode linear index -> (n, do, ho, wo)
    ZYX = DO * HO * WO
    YX = HO * WO

    n = offs // ZYX
    rem = offs % ZYX
    do = rem // YX
    rem = rem % YX
    ho = rem // WO
    wo = rem % WO

    # Cast to int64 for pointer arithmetic
    n64 = n.to(tl.int64)
    do64 = do.to(tl.int64)
    ho64 = ho.to(tl.int64)
    wo64 = wo.to(tl.int64)

    sxn = tl.full([], sxn, tl.int64)
    sxc = tl.full([], sxc, tl.int64)
    sxd = tl.full([], sxd, tl.int64)
    sxh = tl.full([], sxh, tl.int64)
    sxw = tl.full([], sxw, tl.int64)

    syn = tl.full([], syn, tl.int64)
    syc = tl.full([], syc, tl.int64)
    syd = tl.full([], syd, tl.int64)
    syh = tl.full([], syh, tl.int64)
    syw = tl.full([], syw, tl.int64)

    two = tl.full([], 2, tl.int64)
    di0 = do64 * two
    hi0 = ho64 * two
    wi0 = wo64 * two

    # Base pointers for input window start and output
    base_x = n64 * sxn + di0 * sxd + hi0 * sxh + wi0 * sxw
    base_y = n64 * syn + do64 * syd + ho64 * syh + wo64 * syw

    # Offsets for the 2x2x2 pooling window
    o0 = tl.full([], 0, tl.int64)
    o1 = sxw
    o2 = sxh
    o3 = sxh + sxw
    o4 = sxd
    o5 = sxd + sxw
    o6 = sxd + sxh
    o7 = sxd + sxh + sxw

    # Streaming logsumexp across channels in float32
    neg_inf = tl.full((BLOCK,), float("-inf"), tl.float32)
    m = neg_inf
    s = tl.zeros((BLOCK,), dtype=tl.float32)

    p0 = x_ptr + base_x
    # Iterate channels; for each channel compute maxpool(2x2x2) then update running LSE
    for c in tl.static_range(0, C):
        pc = p0 + c * sxc
        v0 = tl.load(pc + o0, mask=mask, other=-float("inf"))
        v1 = tl.load(pc + o1, mask=mask, other=-float("inf"))
        v2 = tl.load(pc + o2, mask=mask, other=-float("inf"))
        v3 = tl.load(pc + o3, mask=mask, other=-float("inf"))
        v4 = tl.load(pc + o4, mask=mask, other=-float("inf"))
        v5 = tl.load(pc + o5, mask=mask, other=-float("inf"))
        v6 = tl.load(pc + o6, mask=mask, other=-float("inf"))
        v7 = tl.load(pc + o7, mask=mask, other=-float("inf"))
        vmax = tl.maximum(v0, v1)
        vmax = tl.maximum(vmax, v2)
        vmax = tl.maximum(vmax, v3)
        vmax = tl.maximum(vmax, v4)
        vmax = tl.maximum(vmax, v5)
        vmax = tl.maximum(vmax, v6)
        vmax = tl.maximum(vmax, v7)

        val = vmax.to(tl.float32)
        new_m = tl.maximum(m, val)
        s = tl.exp(m - new_m) * s + tl.exp(val - new_m)
        m = new_m

    out = tl.log(s) + m
    out = tl.maximum(out, 0.0)
    tl.store(y_ptr + base_y, out, mask=mask)


def _pool_lse_relu_triton(x: torch.Tensor) -> torch.Tensor:
    # x: (N, C, D, H, W)
    assert x.ndim == 5
    if not x.is_cuda:
        # CPU fallback reference
        y = torch.nn.functional.max_pool3d(x, kernel_size=2, stride=2)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = torch.relu(y)
        return y

    if not x.is_contiguous():
        x = x.contiguous()

    N, C, D, H, W = x.shape
    DO, HO, WO = D // 2, H // 2, W // 2
    y = torch.empty((N, 1, DO, HO, WO), device=x.device, dtype=x.dtype)

    sxn, sxc, sxd, sxh, sxw = x.stride()
    syn, syc, syd, syh, syw = y.stride()

    M = N * DO * HO * WO
    grid = lambda META: (triton.cdiv(M, META["BLOCK"]),)
    _pool_lse_relu_stream_kernel[grid](
        x, y,
        M, DO, HO, WO,
        sxn, sxc, sxd, sxh, sxw,
        syn, syc, syd, syh, syw,
        C=C,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
    ],
    key=["M"],
)
@triton.jit
def _lse_relu_reduce_c_kernel(
    x_ptr,            # input pointer (N, C, Z, Y, X)
    out_ptr,          # output pointer (N, 1, Z, Y, X)
    M,                # total number of (N, Z, Y, X) positions
    Z, Y, X,          # spatial dims after pooling
    stride_n,         # strides of input
    stride_c,
    stride_z,
    stride_y,
    stride_x,
    C: tl.constexpr,  # number of channels to reduce over (compile-time)
    BLOCK: tl.constexpr,  # vector width per program
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    # Decode linear index offs -> (n, z, y, x)
    ZYX = Z * Y * X
    YX = Y * X

    n = offs // ZYX
    rem = offs % ZYX
    z = rem // YX
    rem = rem % YX
    y = rem // X
    x = rem % X

    # Cast indices and strides to int64 for pointer arithmetic safety
    n64 = n.to(tl.int64)
    z64 = z.to(tl.int64)
    y64 = y.to(tl.int64)
    x64 = x.to(tl.int64)

    sN = tl.full([], stride_n, tl.int64)
    sC = tl.full([], stride_c, tl.int64)
    sZ = tl.full([], stride_z, tl.int64)
    sY = tl.full([], stride_y, tl.int64)
    sX = tl.full([], stride_x, tl.int64)

    # Base pointer offset for input at channel 0: (n, 0, z, y, x)
    base_in = n64 * sN + z64 * sZ + y64 * sY + x64 * sX
    # Base pointer offset for output (N,1,Z,Y,X) contiguous
    ZYX64 = tl.full([], ZYX, tl.int64)
    YX64 = tl.full([], YX, tl.int64)
    X64 = tl.full([], X, tl.int64)
    base_out = n64 * ZYX64 + z64 * YX64 + y64 * X64 + x64

    # Streaming log-sum-exp across channel dimension in float32
    neg_inf = tl.full((BLOCK,), float("-inf"), tl.float32)
    m = neg_inf  # running max
    s = tl.zeros((BLOCK,), dtype=tl.float32)  # running sum of exp shifted by max

    # Pointer to the first channel and prefetch pipeline
    p = x_ptr + base_in
    if C > 0:
        val = tl.load(p, mask=mask, other=-float("inf"))
        # Unrolled/prefetched channel loop
        for _ in tl.static_range(0, C - 1):
            p_next = p + sC
            val_next = tl.load(p_next, mask=mask, other=-float("inf"))
            # Update running (m, s) with current val
            new_m = tl.maximum(m, val)
            s = tl.exp(m - new_m) * s + tl.exp(val - new_m)
            m = new_m
            # Advance
            p = p_next
            val = val_next
        # Final update for the last prefetched value
        new_m = tl.maximum(m, val)
        s = tl.exp(m - new_m) * s + tl.exp(val - new_m)
        m = new_m

    # Final logsumexp + ReLU in float32
    out_f32 = tl.log(s) + m
    out_f32 = tl.maximum(out_f32, 0.0)

    # Store to output (same spatial/N index but channel dimension is size=1)
    tl.store(out_ptr + base_out, out_f32, mask=mask)


def _lse_relu_triton(x: torch.Tensor) -> torch.Tensor:
    # x shape: (N, C, Z, Y, X)
    assert x.is_cuda, "Triton kernel requires CUDA tensor"

    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()

    N, C, Z, Y, X = x.shape
    out = torch.empty((N, 1, Z, Y, X), device=x.device, dtype=torch.float32)

    # Strides in elements (contiguous guaranteed)
    sN, sC, sZ, sY, sX = x.stride()

    # Total positions over (N, Z, Y, X)
    M = N * Z * Y * X

    grid = lambda META: (triton.cdiv(M, META["BLOCK"]),)
    _lse_relu_reduce_c_kernel[grid](
        x, out,
        M, Z, Y, X,
        sN, sC, sZ, sY, sX,
        C=C,
    )
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    The MaxPool3d + LogSumExp(dim=1, keepdim=True) + ReLU are fused into a single Triton kernel on CUDA.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, depth', height', width')
        """
        x = self.conv(x)
        if x.is_cuda:
            # Fused MaxPool3d -> LogSumExp over channels -> ReLU
            x = _pool_lse_relu_triton(x)
        else:
            x = self.max_pool(x)
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = torch.relu(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]