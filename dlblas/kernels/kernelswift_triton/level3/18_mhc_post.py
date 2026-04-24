import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mhc_post_kernel(
    x_ptr,                     # bfloat16* [n0, n1, h]
    residual_ptr,              # bfloat16* [n0, n1, mhc_mult, h]
    post_ptr,                  # float32*  [n0, n1, mhc_mult, 1]
    comb_ptr,                  # float32*  [n0, n1, mhc_mult, mhc_mult]
    out_ptr,                   # bfloat16* [n0, n1, mhc_mult, h]
    n0, n1, mhc_mult, h,       # sizes
    x_stride_a, x_stride_b, x_stride_c,
    res_stride_a, res_stride_b, res_stride_m, res_stride_c,
    post_stride_a, post_stride_b, post_stride_n, post_stride_c1,
    comb_stride_a, comb_stride_b, comb_stride_m, comb_stride_n,
    out_stride_a, out_stride_b, out_stride_n, out_stride_c,
    BLOCK_N: tl.constexpr,     # tile size along n (mhc_mult)
    BLOCK_C: tl.constexpr,     # tile size along c (h)
    BLOCK_M: tl.constexpr,     # reduction tile along m (mhc_mult)
):
    pid_c = tl.program_id(0)  # tile along c dimension
    pid_b = tl.program_id(1)  # index along b dimension (n1)
    pid_an = tl.program_id(2) # combined a and n-tiles

    tiles_n = tl.cdiv(mhc_mult, BLOCK_N)
    a = pid_an // tiles_n
    n_tile_id = pid_an % tiles_n

    offs_n = n_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_n = offs_n < mhc_mult
    mask_c = offs_c < h

    # Hints for better codegen on contiguous dims
    tl.multiple_of(offs_c, 8)
    tl.multiple_of(offs_n, 4)

    # Load x (a, b, c) as bfloat16 -> float32
    x_ptrs = x_ptr + a * x_stride_a + pid_b * x_stride_b + offs_c * x_stride_c
    x_vals = tl.load(x_ptrs, mask=mask_c, other=0).to(tl.float32)

    # Load post_layer_mix (a, b, n, 0) as float32
    post_ptrs = post_ptr + a * post_stride_a + pid_b * post_stride_b + offs_n * post_stride_n
    post_vals = tl.load(post_ptrs, mask=mask_n, other=0.0)

    # Initialize accumulator with x.float().unsqueeze(-2) * post_layer_mix
    acc = post_vals[:, None] * x_vals[None, :]

    # Reduction over m for term2 = einsum over m: comb_res_mix[a,b,m,n] * residual[a,b,m,c]
    m_start = 0
    while m_start < mhc_mult:
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < mhc_mult

        # Load comb_res_mix tile: shape [M, N]
        comb_ptrs = (
            comb_ptr
            + a * comb_stride_a
            + pid_b * comb_stride_b
            + m_offsets[:, None] * comb_stride_m
            + offs_n[None, :] * comb_stride_n
        )
        comb_vals = tl.load(comb_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # Load residual tile: shape [M, C]
        res_ptrs = (
            residual_ptr
            + a * res_stride_a
            + pid_b * res_stride_b
            + m_offsets[:, None] * res_stride_m
            + offs_c[None, :] * res_stride_c
        )
        res_vals = tl.load(res_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0).to(tl.float32)

        # Accumulate: sum over m of comb[m,n] * residual[m,c] -> [N, C]
        acc += tl.sum(comb_vals[:, :, None] * res_vals[:, None, :], axis=0)

        m_start += BLOCK_M

    # Store result cast to bfloat16
    out_ptrs = (
        out_ptr
        + a * out_stride_a
        + pid_b * out_stride_b
        + offs_n[:, None] * out_stride_n
        + offs_c[None, :] * out_stride_c
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_n[:, None] & mask_c[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # Fallback to reference if not CUDA
        if not x.is_cuda:
            term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
            return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()

        assert x.ndim == 3 and residual.ndim == 4 and post_layer_mix.ndim == 4 and comb_res_mix.ndim == 4
        n0, n1, h = x.shape
        mhc_mult = residual.shape[2]
        # Allocate output
        out = torch.empty((n0, n1, mhc_mult, h), device=x.device, dtype=torch.bfloat16)

        # Strides in elements
        x_sa, x_sb, x_sc = x.stride()
        res_sa, res_sb, res_sm, res_sc = residual.stride()
        post_sa, post_sb, post_sn, post_sc1 = post_layer_mix.stride()
        comb_sa, comb_sb, comb_sm, comb_sn = comb_res_mix.stride()
        out_sa, out_sb, out_sn, out_sc = out.stride()

        # Tiling parameters
        # N (mhc_mult) is typically very small (e.g., 4), choose exact coverage to avoid masked lanes
        BLOCK_N = 4
        # C (h) large, pick 256 for better arithmetic intensity per CTA on Hopper
        BLOCK_C = 256
        # Reduction along M, typically equals mhc_mult; set to 4 and rely on mask otherwise
        BLOCK_M = 4

        grid = (
            triton.cdiv(h, BLOCK_C),                      # tiles along c
            n1,                                           # along b
            n0 * triton.cdiv(mhc_mult, BLOCK_N),          # combined a and n tiles
        )

        mhc_post_kernel[grid](
            x, residual, post_layer_mix, comb_res_mix, out,
            n0, n1, mhc_mult, h,
            x_sa, x_sb, x_sc,
            res_sa, res_sb, res_sm, res_sc,
            post_sa, post_sb, post_sn, post_sc1,
            comb_sa, comb_sb, comb_sm, comb_sn,
            out_sa, out_sb, out_sn, out_sc,
            BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, BLOCK_M=BLOCK_M,
            num_warps=8, num_stages=3,
        )
        return out

# Default test-data sizes
n0=2
n1=4096
h=1280
mhc_mult=4

def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int
) -> dict[str, torch.Tensor]:
    # Generate directly on CUDA for benchmarking the Triton kernel
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device='cuda')
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device='cuda')
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device='cuda')
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device='cuda')

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device='cuda')
    return [x,residual,post_layer_mix,comb_res_mix,o_grad]

def get_inputs():
    x,residual,post_layer_mix,comb_res_mix,o_grad = generate_mhc_post_test_data(n0, n1, h, mhc_mult)
    return [x,residual,post_layer_mix,comb_res_mix]

def get_init_inputs():
    return []