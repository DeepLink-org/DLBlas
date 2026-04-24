import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_einsum_axpy_kernel(
    x_ptr,                     # [a, b, c] bf16
    residual_ptr,              # [a, b, m, c] bf16
    post_layer_mix_ptr,        # [a, b, n, 1] fp32
    comb_res_mix_ptr,          # [a, b, m, n] fp32
    out_ptr,                   # [a, b, n, c] bf16
    n0, n1, C,                 # sizes: a, b, c
    STRIDE_X_A, STRIDE_X_B, STRIDE_X_C,
    STRIDE_R_A, STRIDE_R_B, STRIDE_R_M, STRIDE_R_C,
    STRIDE_P_A, STRIDE_P_B, STRIDE_P_N, STRIDE_P_1,
    STRIDE_CM_A, STRIDE_CM_B, STRIDE_CM_M, STRIDE_CM_N,
    STRIDE_O_A, STRIDE_O_B, STRIDE_O_N, STRIDE_O_C,
    N1,                        # n1 for decoding ab index
    MHC: tl.constexpr,         # mhc_mult (n dimension) - compile-time constant
    BLOCK_C: tl.constexpr,     # tile size along c
):
    pid_ab = tl.program_id(0)
    pid_c = tl.program_id(1)

    a_idx = pid_ab // N1
    b_idx = pid_ab - a_idx * N1

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    # Precompute AB base offsets to reduce repeated integer arithmetic
    x_ab_base = a_idx * STRIDE_X_A + b_idx * STRIDE_X_B
    r_ab_base = a_idx * STRIDE_R_A + b_idx * STRIDE_R_B
    p_ab_base = a_idx * STRIDE_P_A + b_idx * STRIDE_P_B
    cm_ab_base = a_idx * STRIDE_CM_A + b_idx * STRIDE_CM_B
    o_ab_base = a_idx * STRIDE_O_A + b_idx * STRIDE_O_B

    # Load x[a, b, c] as bf16 -> fp32 (streaming, bypass L1)
    x_ptrs = x_ptr + x_ab_base + offs_c * STRIDE_X_C
    x_vec = tl.load(x_ptrs, mask=c_mask, other=0, cache_modifier='.cg').to(tl.float32)

    # Load post_layer_mix[a, b, n] as fp32 and keep in cache
    offs_n = tl.arange(0, MHC)
    plm_ptrs = post_layer_mix_ptr + p_ab_base + offs_n * STRIDE_P_N
    plm_vec = tl.load(plm_ptrs, eviction_policy='evict_last')  # fp32, length MHC

    # Initialize output tile [MHC, BLOCK_C] with x * post_layer_mix
    out_tile = plm_vec[:, None] * x_vec[None, :]  # fp32

    # Software pipeline for m-reduction: prefetch next before using current
    # Prefetch m=0
    res_ptrs = residual_ptr + r_ab_base + offs_c * STRIDE_R_C
    res_vec = tl.load(res_ptrs, mask=c_mask, other=0, cache_modifier='.cg').to(tl.float32)
    crm_ptrs = comb_res_mix_ptr + cm_ab_base + offs_n * STRIDE_CM_N
    crm_vec = tl.load(crm_ptrs, eviction_policy='evict_last')

    for m in range(MHC):
        # FMA: out += crm_vec[:, None] * res_vec[None, :]
        out_tile += crm_vec[:, None] * res_vec[None, :]

        if m + 1 < MHC:
            # Prefetch next m+1
            res_ptrs_next = residual_ptr + r_ab_base + (m + 1) * STRIDE_R_M + offs_c * STRIDE_R_C
            res_vec_next = tl.load(res_ptrs_next, mask=c_mask, other=0, cache_modifier='.cg').to(tl.float32)
            crm_ptrs_next = comb_res_mix_ptr + cm_ab_base + (m + 1) * STRIDE_CM_M + offs_n * STRIDE_CM_N
            crm_vec_next = tl.load(crm_ptrs_next, eviction_policy='evict_last')
            # Advance pipeline
            res_vec = res_vec_next
            crm_vec = crm_vec_next

    # Store result to out[a,b,n,c] as bf16
    out_ptrs = out_ptr + o_ab_base + offs_n[:, None] * STRIDE_O_N + offs_c[None, :] * STRIDE_O_C
    tl.store(out_ptrs, out_tile.to(tl.bfloat16), mask=c_mask[None, :])


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
        # Shapes
        n0, n1, h = x.shape
        mhc_mult = residual.shape[2]
        assert comb_res_mix.shape[:3] == (n0, n1, mhc_mult) and comb_res_mix.shape[3] == mhc_mult
        assert post_layer_mix.shape[:3] == (n0, n1, mhc_mult)

        # Allocate output
        out = torch.empty((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=x.device)

        # Launch Triton kernel only on CUDA; fallback otherwise
        if x.is_cuda:
            # Compute grid
            BLOCK_C = 256
            grid = (n0 * n1, triton.cdiv(h, BLOCK_C))

            # Extract strides (in elements)
            sx0, sx1, sx2 = x.stride()
            sr0, sr1, sr2, sr3 = residual.stride()
            sp0, sp1, sp2, sp3 = post_layer_mix.stride()
            sc0, sc1, sc2, sc3 = comb_res_mix.stride()
            so0, so1, so2, so3 = out.stride()

            fused_einsum_axpy_kernel[grid](
                x, residual, post_layer_mix, comb_res_mix, out,
                n0, n1, h,
                sx0, sx1, sx2,
                sr0, sr1, sr2, sr3,
                sp0, sp1, sp2, sp3,
                sc0, sc1, sc2, sc3,
                so0, so1, so2, so3,
                n1,
                MHC=mhc_mult,
                BLOCK_C=BLOCK_C,
                num_warps=8,
                num_stages=3,
            )
            return out
        else:
            # Fallback: reference implementation on CPU or non-CUDA
            term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
            return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


n0 = 1
n1 = 4096
h = 1280
mhc_mult = 4
device = 'cuda'

def get_inputs():
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    return [
        x, residual, post_layer_mix, comb_res_mix,
    ]

def get_init_inputs():
    return []
