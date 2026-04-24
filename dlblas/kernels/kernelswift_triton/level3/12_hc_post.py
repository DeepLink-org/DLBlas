import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def hc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    y_ptr,
    batch_size,
    seq_len,
    hc_mult,
    hidden_size,
    stride_x_b,
    stride_x_s,
    stride_x_d,
    stride_r_b,
    stride_r_s,
    stride_r_h,
    stride_r_d,
    stride_p_b,
    stride_p_s,
    stride_p_h,
    stride_c_b,
    stride_c_s,
    stride_c_m,
    stride_c_h,
    stride_y_b,
    stride_y_s,
    stride_y_h,
    stride_y_d,
    BLOCK_D: tl.constexpr,
    HC: tl.constexpr,
):
    bs = tl.program_id(0)
    h = tl.program_id(1)

    b = bs // seq_len
    s = bs % seq_len
    offs_d = tl.arange(0, BLOCK_D)

    post_ptrs = post_ptr + b * stride_p_b + s * stride_p_s + h * stride_p_h
    post_val = tl.load(post_ptrs).to(tl.float32)

    for d0 in range(0, hidden_size, BLOCK_D):
        d_offsets = d0 + offs_d
        mask_d = d_offsets < hidden_size

        x_ptrs = x_ptr + b * stride_x_b + s * stride_x_s + d_offsets * stride_x_d
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0).to(tl.float32)

        acc = post_val * x_vals

        for m in tl.static_range(0, HC):
            comb_ptrs = (
                comb_ptr
                + b * stride_c_b
                + s * stride_c_s
                + m * stride_c_m
                + h * stride_c_h
            )
            comb_val = tl.load(comb_ptrs).to(tl.float32)

            residual_ptrs = (
                residual_ptr
                + b * stride_r_b
                + s * stride_r_s
                + m * stride_r_h
                + d_offsets * stride_r_d
            )
            residual_vals = tl.load(residual_ptrs, mask=mask_d, other=0).to(tl.float32)
            acc += comb_val * residual_vals

        y_ptrs = (
            y_ptr
            + b * stride_y_b
            + s * stride_y_s
            + h * stride_y_h
            + d_offsets * stride_y_d
        )
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_d)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        hc_mult = residual.shape[2]

        assert residual.shape == (batch_size, seq_len, hc_mult, hidden_size)
        assert post.shape == (batch_size, seq_len, hc_mult)
        assert comb.shape == (batch_size, seq_len, hc_mult, hc_mult)

        y = torch.empty((batch_size, seq_len, hc_mult, hidden_size), device=x.device, dtype=torch.bfloat16)

        sx0, sx1, sx2 = x.stride()
        sr0, sr1, sr2, sr3 = residual.stride()
        sp0, sp1, sp2 = post.stride()
        sc0, sc1, sc2, sc3 = comb.stride()
        sy0, sy1, sy2, sy3 = y.stride()

        BLOCK_D = 256
        num_warps = 4
        num_stages = 2

        grid = (batch_size * seq_len, hc_mult)
        hc_post_kernel[grid](
            x,
            residual,
            post,
            comb,
            y,
            batch_size,
            seq_len,
            hc_mult,
            hidden_size,
            sx0,
            sx1,
            sx2,
            sr0,
            sr1,
            sr2,
            sr3,
            sp0,
            sp1,
            sp2,
            sc0,
            sc1,
            sc2,
            sc3,
            sy0,
            sy1,
            sy2,
            sy3,
            BLOCK_D=BLOCK_D,
            HC=hc_mult,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y


def generate_test_data(params):
    batch_size = params['batch_size']
    seq_len = params['seq_len']
    hidden_size = params['hidden']
    hc_mult = params['hc']
    x_data = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device='cpu')
    residual_data = torch.randn(batch_size, seq_len, hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    post_data = torch.randn(batch_size, seq_len, hc_mult, dtype=torch.float32, device='cpu')
    comb_data = torch.randn(batch_size, seq_len, hc_mult, hc_mult, dtype=torch.float32, device='cpu')
    o_grad = torch.randn(batch_size, seq_len, hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    return x_data, residual_data, post_data, comb_data, o_grad


def test_hc_post_fwd():
    return ModelNew(*get_init_inputs()).forward(*get_inputs())


def get_inputs():
    params = {'batch_size': 1, 'seq_len': 4096, 'hidden': 1280, 'hc': 4}
    x_data, residual_data, post_data, comb_data, o_grad = generate_test_data(params)
    return [x_data, residual_data, post_data, comb_data]


def get_init_inputs():
    return []
