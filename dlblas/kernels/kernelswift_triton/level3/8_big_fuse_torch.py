import torch
import torch.nn as nn

# Optional: Triton acceleration
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _mhc_pre_norm_fn(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    # residual: [n0, n1, mhc_mult, hidden_size] -> tokens x (mhc_mult*hidden_size)
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    n0, n1 = residual.shape[:2]
    x = residual.flatten(2, 3).float().reshape(n0 * n1, -1)  # [n_tokens, rgs]
    mixes = x @ mhc_fn.T                                      # [n_tokens, mhc_mult3]
    sqrsum = x.square().sum(-1, keepdim=True)                 # [n_tokens, 1]
    mixes = mixes * (sqrsum / x.shape[-1] + mhc_norm_eps).rsqrt()
    return mixes.view(n0, n1, -1)                             # [n0, n1, mhc_mult3]


def _mhc_pre_split_mixes(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = input_mixes.shape[:2]
    scale = torch.cat([
        mhc_scale[0].expand(mhc_mult),
        mhc_scale[1].expand(mhc_mult),
        mhc_scale[2].expand(mhc_mult * mhc_mult),
    ])
    input_mixes = input_mixes * scale + mhc_base
    pre_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_mix = (input_mixes[:, :, mhc_mult:2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_mix = input_mixes[:, :, 2 * mhc_mult:].view(a, b, mhc_mult, mhc_mult)
    return pre_mix, post_mix, comb_mix


def _sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _mhc_pre_apply_mix(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return (x * mix).sum(-2).bfloat16()


# Triton kernels: fused projection + split, and weighted sum
if _TRITON_AVAILABLE:
    @triton.jit
    def mhc_pre_proj_split_kernel(
        residual_ptr,          # bfloat16/float16/float32*, [B, S, mhc_mult, hidden_size]
        weight_ptr,            # float32*, [M, K]
        base_ptr,              # float32*, [M]
        pre_out_ptr,           # float32*, [B, S, mhc_mult, 1]
        post_out_ptr,          # float32*, [B, S, mhc_mult, 1]
        comb_out_ptr,          # float32*, [B, S, mhc_mult, mhc_mult]
        mhc_scale0, mhc_scale1, mhc_scale2,  # float32 scalars
        mhc_post_mult_value,   # float32
        mhc_pre_eps,           # float32
        rms_eps,               # float32
        n1,                    # int32: seq_len
        mhc_mult,              # int32
        hidden_size,           # int32
        M,                     # int32: mhc_mult3 = 2*mhc_mult + mhc_mult*mhc_mult
        K,                     # int32: mhc_mult * hidden_size
        stride_res_b, stride_res_s, stride_res_h, stride_res_g,
        stride_w_m, stride_w_k,
        stride_base,
        stride_pre_b, stride_pre_s, stride_pre_h, stride_pre_1,
        stride_post_b, stride_post_s, stride_post_h, stride_post_1,
        stride_comb_b, stride_comb_s, stride_comb_r, stride_comb_c,
        BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        pid = tl.program_id(0)  # token id across B*S
        b = pid // n1
        s = pid % n1

        # Base pointers for residual and outputs for this token
        res_base = residual_ptr + b * stride_res_b + s * stride_res_s
        pre_base = pre_out_ptr + b * stride_pre_b + s * stride_pre_s
        post_base = post_out_ptr + b * stride_post_b + s * stride_post_s
        comb_base = comb_out_ptr + b * stride_comb_b + s * stride_comb_s

        # -------- Tile 0: fuse RMS pass with first M-tile matvec to reduce residual loads --------
        mo = 0
        m_offsets0 = mo + tl.arange(0, BLOCK_M)
        m_mask0 = m_offsets0 < M
        out0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        sqr_sum = tl.zeros((), dtype=tl.float32)

        k = 0
        while k < K:
            k_offsets = k + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K
            head_off = k_offsets // hidden_size
            hid_off = k_offsets % hidden_size
            x_ptrs = res_base + head_off * stride_res_h + hid_off * stride_res_g
            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Accumulate RMS sum for the token
            sqr_sum += tl.sum(x_vals * x_vals, axis=0)

            # Accumulate matvec for the first M tile
            w_ptrs0 = weight_ptr + m_offsets0[:, None] * stride_w_m + k_offsets[None, :] * stride_w_k
            w_tile0 = tl.load(w_ptrs0, mask=(m_mask0[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)
            out0 += tl.sum(w_tile0 * x_vals[None, :], axis=1)

            k += BLOCK_K

        # Compute RMS scale and apply to first tile
        scale_rms = tl.rsqrt(sqr_sum / K + rms_eps)
        out0 = out0 * scale_rms

        # Apply piecewise scale and base, compute sigmoid once
        s0 = tl.full([BLOCK_M], mhc_scale0, tl.float32)
        s1 = tl.full([BLOCK_M], mhc_scale1, tl.float32)
        s2 = tl.full([BLOCK_M], mhc_scale2, tl.float32)
        cond0_0 = m_offsets0 < mhc_mult
        cond1_0 = (m_offsets0 >= mhc_mult) & (m_offsets0 < 2 * mhc_mult)
        scale_vals0 = tl.where(cond0_0, s0, tl.where(cond1_0, s1, s2))
        out0 = out0 * scale_vals0
        base_ptrs0 = base_ptr + m_offsets0 * stride_base
        base_vals0 = tl.load(base_ptrs0, mask=m_mask0, other=0.0)
        out0 = out0 + base_vals0
        sig0 = tl.sigmoid(out0)

        # Store pre/post/comb for first tile
        pre_mask0 = m_mask0 & (m_offsets0 < mhc_mult)
        pre_vals0 = sig0 + mhc_pre_eps
        pre_ptrs0 = pre_base + m_offsets0 * stride_pre_h
        tl.store(pre_ptrs0, pre_vals0, mask=pre_mask0)

        post_mask0 = m_mask0 & (m_offsets0 >= mhc_mult) & (m_offsets0 < 2 * mhc_mult)
        post_vals0 = sig0 * mhc_post_mult_value
        post_off0 = tl.where(post_mask0, m_offsets0 - mhc_mult, 0)
        post_ptrs0 = post_base + post_off0 * stride_post_h
        tl.store(post_ptrs0, post_vals0, mask=post_mask0)

        comb_mask0 = m_mask0 & (m_offsets0 >= 2 * mhc_mult)
        comb_idx0 = tl.where(comb_mask0, m_offsets0 - 2 * mhc_mult, 0)
        row_idx0 = comb_idx0 // mhc_mult
        col_idx0 = comb_idx0 % mhc_mult
        comb_ptrs0 = comb_base + row_idx0 * stride_comb_r + col_idx0 * stride_comb_c
        tl.store(comb_ptrs0, out0, mask=comb_mask0)

        # -------- Remaining tiles: reuse scale_rms, compute matvec only --------
        mo += BLOCK_M
        while mo < M:
            m_offsets = mo + tl.arange(0, BLOCK_M)
            m_mask = m_offsets < M

            out_seg = tl.zeros([BLOCK_M], dtype=tl.float32)

            k2 = 0
            while k2 < K:
                k_offsets = k2 + tl.arange(0, BLOCK_K)
                k_mask = k_offsets < K

                head_off = k_offsets // hidden_size
                hid_off = k_offsets % hidden_size
                x_ptrs = res_base + head_off * stride_res_h + hid_off * stride_res_g
                x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

                w_ptrs = weight_ptr + m_offsets[:, None] * stride_w_m + k_offsets[None, :] * stride_w_k
                w_tile = tl.load(w_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)
                out_seg += tl.sum(w_tile * x_vals[None, :], axis=1)

                k2 += BLOCK_K

            # Apply RMS scaling computed from first fused pass
            out_seg = out_seg * scale_rms

            # Apply piecewise scale and base
            s0 = tl.full([BLOCK_M], mhc_scale0, tl.float32)
            s1 = tl.full([BLOCK_M], mhc_scale1, tl.float32)
            s2 = tl.full([BLOCK_M], mhc_scale2, tl.float32)
            cond0 = m_offsets < mhc_mult
            cond1 = (m_offsets >= mhc_mult) & (m_offsets < 2 * mhc_mult)
            scale_vals = tl.where(cond0, s0, tl.where(cond1, s1, s2))
            out_seg = out_seg * scale_vals

            base_ptrs = base_ptr + m_offsets * stride_base
            base_vals = tl.load(base_ptrs, mask=m_mask, other=0.0)
            out_seg = out_seg + base_vals

            # Compute sigmoid once for pre/post paths
            sig = tl.sigmoid(out_seg)

            # Store pre
            pre_mask = m_mask & (m_offsets < mhc_mult)
            pre_vals = sig + mhc_pre_eps
            pre_ptrs = pre_base + m_offsets * stride_pre_h
            tl.store(pre_ptrs, pre_vals, mask=pre_mask)

            # Store post
            post_mask = m_mask & (m_offsets >= mhc_mult) & (m_offsets < 2 * mhc_mult)
            post_vals = sig * mhc_post_mult_value
            post_off = tl.where(post_mask, m_offsets - mhc_mult, 0)
            post_ptrs = post_base + post_off * stride_post_h
            tl.store(post_ptrs, post_vals, mask=post_mask)

            # Store comb
            comb_mask = m_mask & (m_offsets >= 2 * mhc_mult)
            comb_idx = tl.where(comb_mask, m_offsets - 2 * mhc_mult, 0)
            row_idx = comb_idx // mhc_mult
            col_idx = comb_idx % mhc_mult
            comb_ptrs = comb_base + row_idx * stride_comb_r + col_idx * stride_comb_c
            tl.store(comb_ptrs, out_seg, mask=comb_mask)

            mo += BLOCK_M

    @triton.jit
    def mhc_weighted_sum_kernel(
        residual_ptr,     # bfloat16/float16/float32*, [B, S, mhc_mult, hidden_size]
        pre_ptr,          # float32*, [B, S, mhc_mult, 1]
        out_ptr,          # bfloat16*, [B, S, hidden_size]
        n1,               # int32
        mhc_mult,         # int32
        hidden_size,      # int32
        stride_res_b, stride_res_s, stride_res_h, stride_res_g,
        stride_pre_b, stride_pre_s, stride_pre_h, stride_pre_1,
        stride_out_b, stride_out_s, stride_out_g,
        BLOCK_D: tl.constexpr,
    ):
        pid_token = tl.program_id(0)  # token across B*S
        pid_col = tl.program_id(1)    # tile along hidden_size

        b = pid_token // n1
        s = pid_token % n1

        # Base pointers for token
        res_base = residual_ptr + b * stride_res_b + s * stride_res_s
        pre_base = pre_ptr + b * stride_pre_b + s * stride_pre_s
        out_base = out_ptr + b * stride_out_b + s * stride_out_s

        d_offsets = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_size

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        h = 0
        while h < mhc_mult:
            # Load scalar weight for this head
            w_ptr = pre_base + h * stride_pre_h
            w = tl.load(w_ptr)  # float32 scalar (the trailing dim is 1)
            # Load residual tile for this head and accumulate
            x_ptrs = res_base + h * stride_res_h + d_offsets * stride_res_g
            x_vals = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            acc += x_vals * w
            h += 1

        # Store accumulated result as bfloat16
        out_ptrs = out_base + d_offsets * stride_out_g
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=d_mask)


class ModelNew(nn.Module):
    """
    Pure PyTorch/Triton-accelerated implementation of the MHC pre-processing fused kernel.

    Pipeline:
      1. RMS-normalized linear projection of residual (mhc_pre_norm_fn) [Triton fused]
      2. Split mixing logits into pre / post / comb components (mhc_pre_split_mixes) [Triton fused]
      3. Sinkhorn doubly-stochastic normalization of comb_mix
      4. Weighted sum of MHC heads with pre_mix to produce layer_input [Triton accelerated]
    """

    def __init__(
        self,
        mhc_mult: int,
        hidden_size: int,
        rms_eps: float = 1e-6,
        mhc_pre_eps: float = 1e-6,
        mhc_sinkhorn_eps: float = 1e-6,
        mhc_post_mult_value: float = 1.0,
        sinkhorn_repeat: int = 10,
    ):
        super().__init__()
        self.mhc_mult = mhc_mult
        self.rms_eps = rms_eps
        self.mhc_pre_eps = mhc_pre_eps
        self.mhc_sinkhorn_eps = mhc_sinkhorn_eps
        self.mhc_post_mult_value = mhc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat

        mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
        self.fn = nn.Parameter(torch.randn(mhc_mult3, mhc_mult * hidden_size) * 1e-4)
        self.mhc_scale = nn.Parameter(torch.randn(3) * 0.1)
        self.mhc_base = nn.Parameter(torch.randn(mhc_mult3) * 0.1)

    def forward(
        self,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            residual: [batch, seq_len, mhc_mult, hidden_size] bfloat16

        Returns:
            post_mix:   [batch, seq_len, mhc_mult, 1]           float32
            comb_mix:   [batch, seq_len, mhc_mult, mhc_mult]    float32
            layer_input:[batch, seq_len, hidden_size]            bfloat16
        """
        B, S, Hm, Hd = residual.shape
        assert Hm == self.mhc_mult, "mhc_mult mismatch with residual shape"
        M = self.mhc_mult * 2 + self.mhc_mult * self.mhc_mult
        K = self.mhc_mult * Hd

        # Allocate outputs
        device = residual.device
        pre_mix = torch.empty((B, S, self.mhc_mult, 1), dtype=torch.float32, device=device)
        post_mix = torch.empty((B, S, self.mhc_mult, 1), dtype=torch.float32, device=device)
        comb_mix = torch.empty((B, S, self.mhc_mult, self.mhc_mult), dtype=torch.float32, device=device)

        if _TRITON_AVAILABLE and residual.is_cuda:
            # Kernel launch configuration
            BLOCK_K = 256
            BLOCK_M = 32

            # Strides
            s_res = residual.stride()
            s_w = self.fn.stride()
            s_base = self.mhc_base.stride()
            s_pre = pre_mix.stride()
            s_post = post_mix.stride()
            s_comb = comb_mix.stride()

            grid = (B * S,)

            mhc_pre_proj_split_kernel[grid](
                residual, self.fn, self.mhc_base,
                pre_mix, post_mix, comb_mix,
                float(self.mhc_scale[0].item()),
                float(self.mhc_scale[1].item()),
                float(self.mhc_scale[2].item()),
                float(self.mhc_post_mult_value),
                float(self.mhc_pre_eps),
                float(self.rms_eps),
                S, self.mhc_mult, Hd, M, K,
                s_res[0], s_res[1], s_res[2], s_res[3],
                s_w[0], s_w[1],
                s_base[0] if len(s_base) > 0 else 1,
                s_pre[0], s_pre[1], s_pre[2], s_pre[3],
                s_post[0], s_post[1], s_post[2], s_post[3],
                s_comb[0], s_comb[1], s_comb[2], s_comb[3],
                BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
                num_warps=4, num_stages=3
            )

            # Sinkhorn normalization (PyTorch)
            comb_mix = _sinkhorn_normalize(comb_mix, repeat=self.sinkhorn_repeat, eps=self.mhc_sinkhorn_eps)

            # Weighted sum kernel for layer_input
            layer_input = torch.empty((B, S, Hd), dtype=torch.bfloat16, device=device)
            s_res = residual.stride()
            s_pre = pre_mix.stride()
            s_out = layer_input.stride()

            grid_ws = (B * S, triton.cdiv(Hd, 128))
            mhc_weighted_sum_kernel[grid_ws](
                residual, pre_mix, layer_input,
                S, self.mhc_mult, Hd,
                s_res[0], s_res[1], s_res[2], s_res[3],
                s_pre[0], s_pre[1], s_pre[2], s_pre[3],
                s_out[0], s_out[1], s_out[2],
                BLOCK_D=128,
                num_warps=2, num_stages=2
            )
        else:
            # Fallback pure PyTorch path (CPU or missing Triton)
            mixes = _mhc_pre_norm_fn(residual, self.fn, None, self.rms_eps)
            pre_mix, post_mix, comb_mix = _mhc_pre_split_mixes(
                mixes, self.mhc_scale, self.mhc_base,
                self.mhc_mult, self.mhc_post_mult_value, self.mhc_pre_eps,
            )
            comb_mix = _sinkhorn_normalize(comb_mix, repeat=self.sinkhorn_repeat, eps=self.mhc_sinkhorn_eps)
            layer_input = _mhc_pre_apply_mix(residual, pre_mix)

        return post_mix, comb_mix, layer_input


n1 = 512
mhc_mult = 4
hidden_size = 1280


def get_inputs():
    residual = torch.randn(1, n1, mhc_mult, hidden_size).bfloat16()
    return [residual]


def get_init_inputs():
    return [mhc_mult, hidden_size]
