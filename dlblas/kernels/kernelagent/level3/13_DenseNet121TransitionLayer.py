import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv1x1_avgpool2_fused_kernel(
    x_ptr,             # float*  [N, Cin, H, W]
    w_ptr,             # float*  [Cout, Cin]
    y_ptr,             # float*  [N, Cout, Ho, Wo]
    N, Cin, H, W, Cout, Ho, Wo,
    sxn, sxc, sxh, sxw,      # strides for x: N, C, H, W
    swo, swi,                 # strides for w: Cout, Cin
    syn, syc, syh, syw,       # strides for y: N, C, H, W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # tile over M = N * Ho * Wo
    pid_n = tl.program_id(axis=1)  # tile over N = Cout

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_M = N * Ho * Wo
    m_mask = m_offs < total_M
    n_mask = n_offs < Cout

    # Decompose flattened M index into (n, ho, wo)
    HWo = Ho * Wo
    n_idx = m_offs // HWo
    rem = m_offs - n_idx * HWo
    ho = rem // Wo
    wo = rem - ho * Wo

    # Convert to int64 for address arithmetic
    n_idx64 = n_idx.to(tl.int64)
    ho64 = ho.to(tl.int64)
    wo64 = wo.to(tl.int64)

    sxn = tl.broadcast_to(tl.full((), sxn, tl.int64), ())
    sxc = tl.broadcast_to(tl.full((), sxc, tl.int64), ())
    sxh = tl.broadcast_to(tl.full((), sxh, tl.int64), ())
    sxw = tl.broadcast_to(tl.full((), sxw, tl.int64), ())

    swo = tl.broadcast_to(tl.full((), swo, tl.int64), ())
    swi = tl.broadcast_to(tl.full((), swi, tl.int64), ())

    syn = tl.broadcast_to(tl.full((), syn, tl.int64), ())
    syc = tl.broadcast_to(tl.full((), syc, tl.int64), ())
    syh = tl.broadcast_to(tl.full((), syh, tl.int64), ())
    syw = tl.broadcast_to(tl.full((), syw, tl.int64), ())

    # Base offsets for the top-left of each 2x2 pooling window
    base_m = n_idx64 * sxn + (2 * ho64) * sxh + (2 * wo64) * sxw  # [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute once
    n_offs64 = n_offs.to(tl.int64)
    mkm = m_mask[:, None]

    # Software pipelining over K with prefetching
    k_arange = tl.arange(0, BLOCK_K)

    # Initial tile
    k0 = 0
    k_init = k0 + k_arange
    k_mask = k_init < Cin
    k_init64 = k_init.to(tl.int64)

    x_offsets = base_m[:, None] + k_init64[None, :] * sxc  # [BM, BK]
    x00 = tl.load(x_ptr + x_offsets, mask=mkm & k_mask[None, :], other=0.0)
    x01 = tl.load(x_ptr + x_offsets + sxw, mask=mkm & k_mask[None, :], other=0.0)
    x10 = tl.load(x_ptr + x_offsets + sxh, mask=mkm & k_mask[None, :], other=0.0)
    x11 = tl.load(x_ptr + x_offsets + sxh + sxw, mask=mkm & k_mask[None, :], other=0.0)
    a_cur = (x00 + x01 + x10 + x11).to(tl.float32) * 0.25  # [BM, BK]

    w_offsets = k_init64[:, None] * swi + n_offs64[None, :] * swo  # [BK, BN]
    # Cache weights aggressively; they are reused across the BM rows
    w_cur = tl.load(w_ptr + w_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.0, cache_modifier=".ca").to(tl.float32)

    k0 += BLOCK_K
    while k0 < Cin:
        k_next = k0 + k_arange
        k_nmask = k_next < Cin
        k_next64 = k_next.to(tl.int64)

        x_offsets_n = base_m[:, None] + k_next64[None, :] * sxc
        x00n = tl.load(x_ptr + x_offsets_n, mask=mkm & k_nmask[None, :], other=0.0)
        x01n = tl.load(x_ptr + x_offsets_n + sxw, mask=mkm & k_nmask[None, :], other=0.0)
        x10n = tl.load(x_ptr + x_offsets_n + sxh, mask=mkm & k_nmask[None, :], other=0.0)
        x11n = tl.load(x_ptr + x_offsets_n + sxh + sxw, mask=mkm & k_nmask[None, :], other=0.0)
        a_next = (x00n + x01n + x10n + x11n).to(tl.float32) * 0.25

        w_offsets_n = k_next64[:, None] * swi + n_offs64[None, :] * swo
        w_next = tl.load(w_ptr + w_offsets_n, mask=k_nmask[:, None] & n_mask[None, :], other=0.0, cache_modifier=".ca").to(tl.float32)

        # Compute on prefetched current tiles
        acc += tl.dot(a_cur, w_cur)

        # Advance
        a_cur = a_next
        w_cur = w_next
        k0 += BLOCK_K

    # Final accumulation
    acc += tl.dot(a_cur, w_cur)

    # Store results to Y
    base_y = n_idx64 * syn + ho64 * syh + wo64 * syw
    y_offsets = base_y[:, None] + n_offs64[None, :] * syc  # [BM, BN]
    tl.store(y_ptr + y_offsets, acc, mask=m_mask[:, None] & n_mask[None, :])


def conv1x1_avgpool2_fused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute AvgPool2d(kernel=2,stride=2) after 1x1 Conv2d (bias=False) by fusing.
    For 1x1 conv without bias, Conv and AvgPool commute exactly, so we implement
    avgpool then conv in one pass: for each 2x2 window, average across spatial,
    then multiply by the 1x1 weights.
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dtype in (torch.float32, torch.float16, torch.bfloat16), "Unsupported dtype"
    N, Cin, H, W = x.shape
    Cout, Cin_w, kh, kw = weight.shape
    assert kh == 1 and kw == 1 and Cin_w == Cin, "Weight must be 1x1 and match input channels"

    Ho = H // 2
    Wo = W // 2

    # Output as fp32 for stable accumulation; cast back if needed
    y = torch.empty((N, Cout, Ho, Wo), device=x.device, dtype=torch.float32)

    # Flatten weight to [Cout, Cin] and ensure contiguous
    w2 = weight.view(Cout, Cin).contiguous()
    x_c = x.contiguous()

    # Tuned launch parameters for H200
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64 if Cin >= 64 else 32

    grid = (
        triton.cdiv(N * Ho * Wo, BLOCK_M),
        triton.cdiv(Cout, BLOCK_N),
    )
    _conv1x1_avgpool2_fused_kernel[grid](
        x_c, w2, y,
        N, Cin, H, W, Cout, Ho, Wo,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        w2.stride(0), w2.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=5,
    )

    if x.dtype != torch.float32:
        return y.to(x.dtype)
    return y


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        # Keep conv module for parameter management and compatibility
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        # Fallback avgpool for CPU or non-CUDA paths
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        x = self.bn(x)
        x = self.relu(x)
        if x.is_cuda and self.conv.weight.is_cuda:
            # Fused Triton kernel: conv1x1 + avgpool2
            return conv1x1_avgpool2_fused(x, self.conv.weight)
        else:
            # Fallback to standard PyTorch path
            return self.avgpool(self.conv(x))


batch_size = 10
num_input_features = 32
num_output_features = 64
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]