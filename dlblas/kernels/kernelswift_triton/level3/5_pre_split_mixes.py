import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mhc_pre_split_fused_kernel(
    x_ptr,            # float32[N, K]
    base_ptr,         # float32[K]
    pre_ptr,          # float32[N, M, 1] viewed as [N, M, 1]
    post_ptr,         # float32[N, M, 1] viewed as [N, M, 1]
    comb_ptr,         # float32[N, M, M]
    scales_ptr,       # float32[3] -> [scale0, scale1, scale2]
    N, M, K,          # int32
    x_stride_n,       # stride for N in x (in elements)
    pre_stride_n, pre_stride_m,   # strides for N and M in pre (in elements)
    post_stride_n, post_stride_m, # strides for N and M in post (in elements)
    comb_stride_n, comb_stride_m, comb_stride_k,  # strides for N, M, K in comb (in elements)
    post_mult,        # float32
    pre_eps,          # float32
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Row pointers
    x_row_ptr = x_ptr + pid_n * x_stride_n
    pre_row_ptr = pre_ptr + pid_n * pre_stride_n
    post_row_ptr = post_ptr + pid_n * post_stride_n
    comb_row_ptr = comb_ptr + pid_n * comb_stride_n

    offs = pid_t * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < K

    # Load inputs and per-channel base
    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
    base = tl.load(base_ptr + offs, mask=mask, other=0.0)

    # Load scales
    s0 = tl.load(scales_ptr + 0)
    s1 = tl.load(scales_ptr + 1)
    s2 = tl.load(scales_ptr + 2)

    # Tile boundaries and helpers
    tile_start = pid_t * BLOCK_K
    tile_end = tile_start + BLOCK_K
    M2 = M + M

    # Specialize for tiles fully contained in a single logical region to avoid unnecessary work
    if tile_end <= M:
        # Entirely in pre region
        y = x * s0 + base
        ysig = 1.0 / (1.0 + tl.exp(-y))
        tl.store(pre_row_ptr + offs * pre_stride_m, ysig + pre_eps, mask=mask)
    elif (tile_start >= M) & (tile_end <= M2):
        # Entirely in post region
        y = x * s1 + base
        ysig = 1.0 / (1.0 + tl.exp(-y))
        idx1 = offs - M
        tl.store(post_row_ptr + idx1 * post_stride_m, ysig * post_mult, mask=mask)
    elif tile_start >= M2:
        # Entirely in comb region
        y = x * s2 + base
        idx2 = offs - M2
        # Fast path for contiguous [M, M] tiles
        if (comb_stride_k == 1) & (comb_stride_m == M):
            tl.store(comb_row_ptr + idx2, y, mask=mask)
        else:
            i = idx2 // M
            j = idx2 - i * M
            tl.store(comb_row_ptr + i * comb_stride_m + j * comb_stride_k, y, mask=mask)
    else:
        # General case: tile overlaps multiple regions
        scale = tl.where(offs < M, s0, tl.where(offs < M2, s1, s2))
        y = x * scale + base
        ysig = 1.0 / (1.0 + tl.exp(-y))

        # Region masks
        cond0 = (offs < M) & mask
        cond1 = (offs >= M) & (offs < M2) & mask
        cond2 = (offs >= M2) & mask

        # Write pre_mix: sigmoid + eps -> [N, M, 1]
        tl.store(pre_row_ptr + offs * pre_stride_m, ysig + pre_eps, mask=cond0)

        # Write post_mix: sigmoid * post_mult -> [N, M, 1]
        idx1 = offs - M
        tl.store(post_row_ptr + idx1 * post_stride_m, ysig * post_mult, mask=cond1)

        # Write comb_mix: raw y -> [N, M, M]
        idx2 = offs - M2
        # Fast path for contiguous [M, M] tiles
        if (comb_stride_k == 1) & (comb_stride_m == M):
            tl.store(comb_row_ptr + idx2, y, mask=cond2)
        else:
            i = idx2 // M
            j = idx2 - i * M
            tl.store(comb_row_ptr + i * comb_stride_m + j * comb_stride_k, y, mask=cond2)


class ModelNew(nn.Module):
    """
    Triton-optimized implementation of mhc_pre_split_mixes.
    Applies per-channel scale + bias to input_mixes, then splits into:
      - pre_mix:  sigmoid(x[:mhc_mult])        + mhc_pre_eps  -> [*, mhc_mult, 1]
      - post_mix: sigmoid(x[mhc_mult:2*mhc_mult]) * mhc_post_mult_value -> [*, mhc_mult, 1]
      - comb_mix: x[2*mhc_mult:].view(*, mhc_mult, mhc_mult)
    """
    def __init__(
        self,
        mhc_mult: int,
        mhc_post_mult_value: float = 2.0,
        mhc_pre_eps: float = 1e-2,
    ):
        super().__init__()
        self.mhc_mult = mhc_mult
        self.mhc_post_mult_value = mhc_post_mult_value
        self.mhc_pre_eps = mhc_pre_eps
        mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
        self.mhc_scale = nn.Parameter(torch.randn(3) * 0.1)
        self.mhc_base = nn.Parameter(torch.randn(mhc_mult3) * 0.1)

    def forward(
        self,
        input_mixes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_mixes: [batch, seq_len, mhc_mult3] float32
        Returns:
            pre_mix:  [batch, seq_len, mhc_mult, 1]           float32
            post_mix: [batch, seq_len, mhc_mult, 1]           float32
            comb_mix: [batch, seq_len, mhc_mult, mhc_mult]    float32
        """
        m = self.mhc_mult
        a, b = input_mixes.shape[:2]
        k = input_mixes.shape[-1]

        # Fallback to PyTorch for non-CUDA or non-fp32
        if (not input_mixes.is_cuda) or (input_mixes.dtype != torch.float32):
            scale = torch.cat([
                self.mhc_scale[0].expand(m),
                self.mhc_scale[1].expand(m),
                self.mhc_scale[2].expand(m * m),
            ])
            x = input_mixes * scale + self.mhc_base
            pre_mix = x[:, :, :m].sigmoid().unsqueeze(-1) + self.mhc_pre_eps
            post_mix = (x[:, :, m:2 * m].sigmoid() * self.mhc_post_mult_value).unsqueeze(-1)
            comb_mix = x[:, :, 2 * m:].view(a, b, m, m)
            return pre_mix, post_mix, comb_mix

        # CUDA Triton path
        N = a * b
        # Allocate outputs
        pre_mix = torch.empty((a, b, m, 1), device=input_mixes.device, dtype=input_mixes.dtype)
        post_mix = torch.empty((a, b, m, 1), device=input_mixes.device, dtype=input_mixes.dtype)
        comb_mix = torch.empty((a, b, m, m), device=input_mixes.device, dtype=input_mixes.dtype)

        # Views for simpler stride calculations
        x2 = input_mixes.view(N, k)
        pre2 = pre_mix.view(N, m, 1)
        post2 = post_mix.view(N, m, 1)
        comb2 = comb_mix.view(N, m, m)

        # Ensure scales and base are on the same device
        scales = self.mhc_scale
        base = self.mhc_base

        # Tune tile size to enable region-specialized tiles while keeping good occupancy
        BLOCK_K = 16
        grid = (N, triton.cdiv(k, BLOCK_K))

        mhc_pre_split_fused_kernel[grid](
            x2, base, pre2, post2, comb2, scales,
            N, m, k,
            x2.stride(0),
            pre2.stride(0), pre2.stride(1),
            post2.stride(0), post2.stride(1),
            comb2.stride(0), comb2.stride(1), comb2.stride(2),
            self.mhc_post_mult_value, self.mhc_pre_eps,
            BLOCK_K=BLOCK_K,
            num_warps=2,
            num_stages=1,
        )
        return pre_mix, post_mix, comb_mix

n0 = 1
n1 = 1024
mhc_mult = 4
def get_inputs():
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    input_mixes = torch.randn(n0, n1, mhc_mult3)
    return [input_mixes]
def get_init_inputs():
    return [mhc_mult]