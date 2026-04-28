import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mix_residual_kernel(
    x_ptr,                   # *bfloat16  [B, S, D]
    residual_ptr,            # *bfloat16  [B, S, M, D]
    post_ptr,                # *float32   [B, S, M, 1]
    comb_ptr,                # *float32   [B, S, M, M]  (indexing: [in, out])
    y_ptr,                   # *bfloat16  [B, S, M, D]
    B, S, D,
    stride_x_b, stride_x_s, stride_x_d,
    stride_res_b, stride_res_s, stride_res_m, stride_res_d,
    stride_post_b, stride_post_s, stride_post_m, stride_post_one,
    stride_comb_b, stride_comb_s, stride_comb_in, stride_comb_out,
    stride_y_b, stride_y_s, stride_y_m, stride_y_d,
    M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bs // S
    s = pid_bs % S

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Base pointers for (b, s)
    x_bs_ptr = x_ptr + b * stride_x_b + s * stride_x_s
    res_bs_ptr = residual_ptr + b * stride_res_b + s * stride_res_s
    post_bs_ptr = post_ptr + b * stride_post_b + s * stride_post_s
    comb_bs_ptr = comb_ptr + b * stride_comb_b + s * stride_comb_s
    y_bs_ptr = y_ptr + b * stride_y_b + s * stride_y_s

    # Load x slice [D] and cast to f32
    x_vec = tl.load(x_bs_ptr + d_offsets * stride_x_d, mask=d_mask, other=0.0)
    x_vec_f = x_vec.to(tl.float32)

    # For each output head j, compute:
    # y[b, s, j, d] = post[b, s, j, 0] * x[b, s, d]
    #                 + sum_i comb[b, s, i, j] * residual[b, s, i, d]
    for j in tl.static_range(0, M):
        # post scalar
        post_val = tl.load(post_bs_ptr + j * stride_post_m + 0 * stride_post_one)
        # initialize accumulator with post * x
        acc = x_vec_f * post_val

        # accumulate over input heads i
        for i in tl.static_range(0, M):
            res_vec = tl.load(
                res_bs_ptr + i * stride_res_m + d_offsets * stride_res_d,
                mask=d_mask,
                other=0.0,
            )
            res_vec_f = res_vec.to(tl.float32)
            comb_val = tl.load(comb_bs_ptr + i * stride_comb_in + j * stride_comb_out)
            acc = acc + res_vec_f * comb_val

        # store result
        tl.store(y_bs_ptr + j * stride_y_m + d_offsets * stride_y_d, acc.to(tl.bfloat16), mask=d_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: torch.Tensor,                # [B, S, D], bfloat16
        residual: torch.Tensor,         # [B, S, M, D], bfloat16
        post_layer_mix: torch.Tensor,   # [B, S, M, 1], float32
        comb_res_mix: torch.Tensor,     # [B, S, M, M], float32
    ) -> torch.Tensor:
        B, S, D = x.shape
        M = residual.shape[2]

        # Output tensor y: [B, S, M, D], same dtype as x after final cast
        y = torch.empty((B, S, M, D), dtype=x.dtype, device=x.device)

        # Prepare strides (element-wise, not bytes)
        sx_b, sx_s, sx_d = x.stride()
        sr_b, sr_s, sr_m, sr_d = residual.stride()
        sp_b, sp_s, sp_m, sp_one = post_layer_mix.stride()
        sc_b, sc_s, sc_in, sc_out = comb_res_mix.stride()
        sy_b, sy_s, sy_m, sy_d = y.stride()

        # Tile along D; fuse B and S into a single grid dim
        BLOCK_D = 256
        grid = (B * S, triton.cdiv(D, BLOCK_D))

        _mix_residual_kernel[grid](
            x, residual, post_layer_mix, comb_res_mix, y,
            B, S, D,
            sx_b, sx_s, sx_d,
            sr_b, sr_s, sr_m, sr_d,
            sp_b, sp_s, sp_m, sp_one,
            sc_b, sc_s, sc_in, sc_out,
            sy_b, sy_s, sy_m, sy_d,
            M=M, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return y

# Problem setup (unchanged)
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