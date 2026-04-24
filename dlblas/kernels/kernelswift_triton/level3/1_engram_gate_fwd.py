import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def engram_gate_fwd_kernel(
    x_ptr, k_ptr, v_ptr, wh_ptr, we_ptr,
    y_ptr,
    rawdot_ptr, gate_ptr, rstdx_ptr, rstdk_ptr,
    num_tokens, hc_mult, hidden_size,
    stride_x_t, stride_x_h, stride_x_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_d,
    stride_w_h, stride_w_d,
    stride_y_t, stride_y_h, stride_y_d,
    stride_rd_t, stride_rd_h,
    stride_gt_t, stride_gt_h,
    stride_rx_t, stride_rx_h,
    stride_rk_t, stride_rk_h,
    eps, clamp_value, scalar,
    BLOCK_D: tl.constexpr,
):
    # Program IDs along token and head axes
    t = tl.program_id(0)
    h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)

    # Precompute base pointers to reduce pointer arithmetic in the loop
    x_th = x_ptr + t * stride_x_t + h * stride_x_h
    k_th = k_ptr + t * stride_k_t + h * stride_k_h
    wh_h = wh_ptr + h * stride_w_h
    we_h = we_ptr + h * stride_w_h

    # Accumulators for reductions over hidden dimension D
    sxx = tl.zeros((), dtype=tl.float32)
    skk = tl.zeros((), dtype=tl.float32)
    sdot = tl.zeros((), dtype=tl.float32)

    # Double-buffered reduction across D for better latency hiding
    # Preload tile 0
    d_init = offs_d
    mask_cur = d_init < hidden_size
    x_cur_bf = tl.load(x_th + d_init * stride_x_d, mask=mask_cur, other=0.0)
    k_cur_bf = tl.load(k_th + d_init * stride_k_d, mask=mask_cur, other=0.0)
    wh_cur_bf = tl.load(wh_h + d_init * stride_w_d, mask=mask_cur, other=0.0)
    we_cur_bf = tl.load(we_h + d_init * stride_w_d, mask=mask_cur, other=0.0)

    for d0 in range(0, hidden_size, BLOCK_D):
        # Prefetch next tile early
        d_next = d0 + BLOCK_D + offs_d
        mask_next = d_next < hidden_size
        x_next_bf = tl.load(x_th + d_next * stride_x_d, mask=mask_next, other=0.0)
        k_next_bf = tl.load(k_th + d_next * stride_k_d, mask=mask_next, other=0.0)
        wh_next_bf = tl.load(wh_h + d_next * stride_w_d, mask=mask_next, other=0.0)
        we_next_bf = tl.load(we_h + d_next * stride_w_d, mask=mask_next, other=0.0)

        # Compute on current tile
        x_f = x_cur_bf.to(tl.float32)
        k_f = k_cur_bf.to(tl.float32)
        wh_f = wh_cur_bf.to(tl.float32)
        we_f = we_cur_bf.to(tl.float32)

        sxx += tl.sum(x_f * x_f, axis=0)
        skk += tl.sum(k_f * k_f, axis=0)
        # raw dot: sum(x * k * wh * we)
        sdot += tl.sum((x_f * k_f) * (wh_f * we_f), axis=0)

        # Rotate buffers
        x_cur_bf = x_next_bf
        k_cur_bf = k_next_bf
        wh_cur_bf = wh_next_bf
        we_cur_bf = we_next_bf

    D_f = tl.full((), hidden_size, dtype=tl.float32)
    mean_x2 = sxx / D_f
    mean_k2 = skk / D_f
    rstd_x = tl.rsqrt(mean_x2 + eps)
    rstd_k = tl.rsqrt(mean_k2 + eps)
    raw_dot = sdot
    dot = raw_dot * rstd_x * rstd_k * scalar

    # signed_sqrt = sqrt(max(abs(dot), clamp_value)) * sign(dot) with sign(0)=0 to match torch.sign
    abs_dot = tl.abs(dot)
    clamped = tl.maximum(abs_dot, clamp_value)
    sign = tl.where(dot > 0, 1.0, tl.where(dot < 0, -1.0, 0.0))
    signed_sqrt = tl.sqrt(clamped) * sign
    gate = tl.sigmoid(signed_sqrt)

    # Store per-(t,h) scalars
    tl.store(rawdot_ptr + t * stride_rd_t + h * stride_rd_h, raw_dot)
    tl.store(gate_ptr + t * stride_gt_t + h * stride_gt_h, gate)
    tl.store(rstdx_ptr + t * stride_rx_t + h * stride_rx_h, rstd_x)
    tl.store(rstdk_ptr + t * stride_rk_t + h * stride_rk_h, rstd_k)

    # Phase 2: write output y = x + gate * v (also double-buffered)
    v_t = v_ptr + t * stride_v_t
    y_th = y_ptr + t * stride_y_t + h * stride_y_h

    # Preload tile 0 for x and v
    d_init = offs_d
    mask_cur = d_init < hidden_size
    x_cur_bf = tl.load(x_th + d_init * stride_x_d, mask=mask_cur, other=0.0)
    v_cur_bf = tl.load(v_t + d_init * stride_v_d, mask=mask_cur, other=0.0)

    for d0 in range(0, hidden_size, BLOCK_D):
        # Prefetch next tile early
        d_next = d0 + BLOCK_D + offs_d
        mask_next = d_next < hidden_size
        x_next_bf = tl.load(x_th + d_next * stride_x_d, mask=mask_next, other=0.0)
        v_next_bf = tl.load(v_t + d_next * stride_v_d, mask=mask_next, other=0.0)

        # Compute on current tile
        x_f = x_cur_bf.to(tl.float32)
        v_f = v_cur_bf.to(tl.float32)
        y_f = x_f + gate * v_f
        y_bf = y_f.to(tl.bfloat16)
        # Store current tile
        mask_store = (d0 + offs_d) < hidden_size
        tl.store(y_th + (d0 + offs_d) * stride_y_d, y_bf, mask=mask_store)

        # Rotate buffers
        x_cur_bf = x_next_bf
        v_cur_bf = v_next_bf


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weight_hidden: torch.Tensor,
        weight_embed: torch.Tensor,
        clamp_value: float,
        eps: float,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch/Triton fused implementation of engram gate.
        Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v
        Returns (output_bf16, raw_dot_f32, gate_score_f32, rstd_x_f32, rstd_k_f32)
        """
        # Fallback to PyTorch if not on CUDA
        if not hidden_states.is_cuda:
            hidden_size = hidden_states.shape[-1]
            scalar = hidden_size**-0.5
            x = hidden_states.float()
            k_f = k.float()
            wh = weight_hidden.float().unsqueeze(0)
            we = weight_embed.float().unsqueeze(0)
            rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
            rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)
            raw_dot = torch.einsum('...d,...d->...', x * wh, k_f * we)
            dot = raw_dot * rstd_x * rstd_k * scalar
            signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
            gate_score = signed_sqrt.sigmoid()
            output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
            output = output.bfloat16()
            return output, raw_dot, gate_score, rstd_x, rstd_k

        # Triton path
        assert hidden_states.dtype == torch.bfloat16
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16
        assert weight_hidden.dtype == torch.bfloat16
        assert weight_embed.dtype == torch.bfloat16

        T, HC, D = hidden_states.shape
        scalar = float(D ** -0.5)

        # Allocate outputs
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        raw_dot = torch.empty((T, HC), dtype=torch.float32, device=hidden_states.device)
        gate_score = torch.empty((T, HC), dtype=torch.float32, device=hidden_states.device)
        rstd_x = torch.empty((T, HC), dtype=torch.float32, device=hidden_states.device)
        rstd_k = torch.empty((T, HC), dtype=torch.float32, device=hidden_states.device)

        # Strides (in elements)
        sx0, sx1, sx2 = hidden_states.stride()
        sk0, sk1, sk2 = k.stride()
        sv0, sv1 = v.stride()
        sw0, sw1 = weight_hidden.stride()
        sy0, sy1, sy2 = y.stride()
        srd0, srd1 = raw_dot.stride()
        sgt0, sgt1 = gate_score.stride()
        srx0, srx1 = rstd_x.stride()
        srk0, srk1 = rstd_k.stride()

        # Tuned block configuration for H100/H200-class GPUs
        BLOCK_D = 1024
        num_warps = 8
        num_stages = 4

        grid = (T, HC)
        engram_gate_fwd_kernel[grid](
            hidden_states, k, v, weight_hidden, weight_embed,
            y,
            raw_dot, gate_score, rstd_x, rstd_k,
            T, HC, D,
            sx0, sx1, sx2,
            sk0, sk1, sk2,
            sv0, sv1,
            sw0, sw1,
            sy0, sy1, sy2,
            srd0, srd1,
            sgt0, sgt1,
            srx0, srx1,
            srk0, srk1,
            float(eps), float(clamp_value), float(scalar),
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y, raw_dot, gate_score, rstd_x, rstd_k


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hc_mult = params['hc']
    hidden_size = params['hidden']
    eps = 1e-20
    clamp_value = 1e-6
    x_data = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    k_data = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    v_data = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cpu')
    wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    weight_fused = wh_data.float() * we_data.float()
    return x_data, k_data, v_data, wh_data, we_data, weight_fused, eps, clamp_value


def test_engram_gate_fwd():
    return Model(*get_init_inputs()).forward(*get_inputs())


def get_inputs():
    params = {'num_tokens': 4096, 'hc': 4, 'hidden': 4096}
    x_data, k_data, v_data, wh_data, we_data, weight_fused, eps, clamp_value = generate_test_data(params)
    return [x_data, k_data, v_data, wh_data, we_data, clamp_value, eps]


def get_init_inputs():
    return []