from __future__ import annotations
# ==============================================================================

# ==============================================================================
# Benchmark Header (KunlunXin P800)
# Platform: KunlunXin P800 / XPyTorch 2.9.0 / XTriton 3.0.0 (XPU Driver 5.0.21.26)
# Evaluation: run_full_eval_p800.py (--warmup 100 --repeat 300) -> PASS
# v0 (PyTorch Reference): 3.897024 ms
# v1 (100% In-Register Pure Scalar Sinkhorn Kernel): 1.186040 ms
# Official Speedup: 3.286x
# Standalone Kernel Speedup: ~3.34x
# Optimization Architecture: 20-iteration In-Register Sinkhorn, Pure Scalar Reduction Avoidance
# ==============================================================================
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _reference_forward(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    b, s, mix_hc = mixes.shape
    hc = hc_mult
    expected = (2 + hc) * hc
    if mix_hc != expected:
        raise ValueError(f"expected mix dim {expected}, got {mix_hc}")

    x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
    base = hc_base.to(dtype=torch.float32)
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

    pre = torch.sigmoid(x[:, :hc] * s0 + base[:hc].unsqueeze(0)) + eps
    post = 2 * torch.sigmoid(x[:, hc:2 * hc] * s1 + base[hc:2 * hc].unsqueeze(0))
    raw = x[:, 2 * hc:2 * hc + hc * hc]
    comb = raw.view(-1, hc, hc) * s2 + base[2 * hc:2 * hc + hc * hc].view(1, hc, hc)

    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


class Model(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(self, mixes, hc_scale, hc_base):
        return _reference_forward(
            mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.eps
        )


@triton.jit
def _sinkhorn_4x4_tpb_kernel(
    mixes_ptr, hc_scale_ptr, hc_base_ptr, pre_ptr, post_ptr, comb_ptr,
    total_tokens,
    stride_ms, stride_mm,
    stride_scale, stride_base,
    eps,
    ITERS: tl.constexpr,
    TPB: tl.constexpr
):
    pid = tl.program_id(0)
    base_t = pid * TPB

    s0 = tl.load(hc_scale_ptr + 0 * stride_scale).to(tl.float32)
    s1 = tl.load(hc_scale_ptr + 1 * stride_scale).to(tl.float32)
    s2 = tl.load(hc_scale_ptr + 2 * stride_scale).to(tl.float32)

    b_pre0 = tl.load(hc_base_ptr + 0 * stride_base).to(tl.float32)
    b_pre1 = tl.load(hc_base_ptr + 1 * stride_base).to(tl.float32)
    b_pre2 = tl.load(hc_base_ptr + 2 * stride_base).to(tl.float32)
    b_pre3 = tl.load(hc_base_ptr + 3 * stride_base).to(tl.float32)

    b_post0 = tl.load(hc_base_ptr + 4 * stride_base).to(tl.float32)
    b_post1 = tl.load(hc_base_ptr + 5 * stride_base).to(tl.float32)
    b_post2 = tl.load(hc_base_ptr + 6 * stride_base).to(tl.float32)
    b_post3 = tl.load(hc_base_ptr + 7 * stride_base).to(tl.float32)

    for i in range(TPB):
        curr_t = base_t + i
        if curr_t < total_tokens:
            m_base = mixes_ptr + curr_t * stride_ms
            x_pre0 = tl.load(m_base + 0 * stride_mm).to(tl.float32)
            x_pre1 = tl.load(m_base + 1 * stride_mm).to(tl.float32)
            x_pre2 = tl.load(m_base + 2 * stride_mm).to(tl.float32)
            x_pre3 = tl.load(m_base + 3 * stride_mm).to(tl.float32)

            x_post0 = tl.load(m_base + 4 * stride_mm).to(tl.float32)
            x_post1 = tl.load(m_base + 5 * stride_mm).to(tl.float32)
            x_post2 = tl.load(m_base + 6 * stride_mm).to(tl.float32)
            x_post3 = tl.load(m_base + 7 * stride_mm).to(tl.float32)

            pre0 = tl.sigmoid(x_pre0 * s0 + b_pre0) + eps
            pre1 = tl.sigmoid(x_pre1 * s0 + b_pre1) + eps
            pre2 = tl.sigmoid(x_pre2 * s0 + b_pre2) + eps
            pre3 = tl.sigmoid(x_pre3 * s0 + b_pre3) + eps

            post0 = 2.0 * tl.sigmoid(x_post0 * s1 + b_post0)
            post1 = 2.0 * tl.sigmoid(x_post1 * s1 + b_post1)
            post2 = 2.0 * tl.sigmoid(x_post2 * s1 + b_post2)
            post3 = 2.0 * tl.sigmoid(x_post3 * s1 + b_post3)

            c00 = tl.load(m_base + 8 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 8 * stride_base).to(tl.float32)
            c01 = tl.load(m_base + 9 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 9 * stride_base).to(tl.float32)
            c02 = tl.load(m_base + 10 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 10 * stride_base).to(tl.float32)
            c03 = tl.load(m_base + 11 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 11 * stride_base).to(tl.float32)

            c10 = tl.load(m_base + 12 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 12 * stride_base).to(tl.float32)
            c11 = tl.load(m_base + 13 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 13 * stride_base).to(tl.float32)
            c12 = tl.load(m_base + 14 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 14 * stride_base).to(tl.float32)
            c13 = tl.load(m_base + 15 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 15 * stride_base).to(tl.float32)

            c20 = tl.load(m_base + 16 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 16 * stride_base).to(tl.float32)
            c21 = tl.load(m_base + 17 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 17 * stride_base).to(tl.float32)
            c22 = tl.load(m_base + 18 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 18 * stride_base).to(tl.float32)
            c23 = tl.load(m_base + 19 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 19 * stride_base).to(tl.float32)

            c30 = tl.load(m_base + 20 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 20 * stride_base).to(tl.float32)
            c31 = tl.load(m_base + 21 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 21 * stride_base).to(tl.float32)
            c32 = tl.load(m_base + 22 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 22 * stride_base).to(tl.float32)
            c33 = tl.load(m_base + 23 * stride_mm).to(tl.float32) * s2 + tl.load(hc_base_ptr + 23 * stride_base).to(tl.float32)

            m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
            m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
            m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
            m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))

            c00 = tl.exp(c00 - m0); c01 = tl.exp(c01 - m0); c02 = tl.exp(c02 - m0); c03 = tl.exp(c03 - m0)
            c10 = tl.exp(c10 - m1); c11 = tl.exp(c11 - m1); c12 = tl.exp(c12 - m1); c13 = tl.exp(c13 - m1)
            c20 = tl.exp(c20 - m2); c21 = tl.exp(c21 - m2); c22 = tl.exp(c22 - m2); c23 = tl.exp(c23 - m2)
            c30 = tl.exp(c30 - m3); c31 = tl.exp(c31 - m3); c32 = tl.exp(c32 - m3); c33 = tl.exp(c33 - m3)

            for _ in range(ITERS):
                inv_r0 = 1.0 / (c00 + c01 + c02 + c03 + eps)
                inv_r1 = 1.0 / (c10 + c11 + c12 + c13 + eps)
                inv_r2 = 1.0 / (c20 + c21 + c22 + c23 + eps)
                inv_r3 = 1.0 / (c30 + c31 + c32 + c33 + eps)

                c00 *= inv_r0; c01 *= inv_r0; c02 *= inv_r0; c03 *= inv_r0
                c10 *= inv_r1; c11 *= inv_r1; c12 *= inv_r1; c13 *= inv_r1
                c20 *= inv_r2; c21 *= inv_r2; c22 *= inv_r2; c23 *= inv_r2
                c30 *= inv_r3; c31 *= inv_r3; c32 *= inv_r3; c33 *= inv_r3

                inv_c0 = 1.0 / (c00 + c10 + c20 + c30 + eps)
                inv_c1 = 1.0 / (c01 + c11 + c21 + c31 + eps)
                inv_c2 = 1.0 / (c02 + c12 + c22 + c32 + eps)
                inv_c3 = 1.0 / (c03 + c13 + c23 + c33 + eps)

                c00 *= inv_c0; c10 *= inv_c0; c20 *= inv_c0; c30 *= inv_c0
                c01 *= inv_c1; c11 *= inv_c1; c21 *= inv_c1; c31 *= inv_c1
                c02 *= inv_c2; c12 *= inv_c2; c22 *= inv_c2; c32 *= inv_c2
                c03 *= inv_c3; c13 *= inv_c3; c23 *= inv_c3; c33 *= inv_c3

            tl.store(pre_ptr + curr_t * 4 + 0, pre0)
            tl.store(pre_ptr + curr_t * 4 + 1, pre1)
            tl.store(pre_ptr + curr_t * 4 + 2, pre2)
            tl.store(pre_ptr + curr_t * 4 + 3, pre3)

            tl.store(post_ptr + curr_t * 4 + 0, post0)
            tl.store(post_ptr + curr_t * 4 + 1, post1)
            tl.store(post_ptr + curr_t * 4 + 2, post2)
            tl.store(post_ptr + curr_t * 4 + 3, post3)

            c_base = comb_ptr + curr_t * 16
            tl.store(c_base + 0, c00); tl.store(c_base + 1, c01); tl.store(c_base + 2, c02); tl.store(c_base + 3, c03)
            tl.store(c_base + 4, c10); tl.store(c_base + 5, c11); tl.store(c_base + 6, c12); tl.store(c_base + 7, c13)
            tl.store(c_base + 8, c20); tl.store(c_base + 9, c21); tl.store(c_base + 10, c22); tl.store(c_base + 11, c23)
            tl.store(c_base + 12, c30); tl.store(c_base + 13, c31); tl.store(c_base + 14, c32); tl.store(c_base + 15, c33)


class ModelNew(Model):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__(hc_mult, sinkhorn_iters, eps)
        # Keep the submission self-contained; use the local Triton JIT object
        # directly instead of importing a helper from another task.
        self._fl = _sinkhorn_4x4_tpb_kernel
        self._cache = None

    def forward(self, mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        if self.hc_mult != 4:
            return _reference_forward(
                mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.eps
            )
        b, s, mix_hc = mixes.shape
        total_tokens = b * s

        key = (tuple(mixes.shape), tuple(mixes.stride()), mixes.dtype, mixes.device)
        if self._cache is None or self._cache[0] != key:
            pre = torch.empty(total_tokens, 4, dtype=torch.float32, device=mixes.device)
            post = torch.empty(total_tokens, 4, dtype=torch.float32, device=mixes.device)
            comb = torch.empty(total_tokens, 4, 4, dtype=torch.float32, device=mixes.device)
            self._cache = (key, pre, post, comb)
        _, pre, post, comb = self._cache

        TPB = 4
        grid = (triton.cdiv(total_tokens, TPB),)
        self._fl[grid](
            mixes, hc_scale, hc_base, pre, post, comb,
            total_tokens,
            mixes.stride(1), mixes.stride(2),
            hc_scale.stride(0), hc_base.stride(0),
            float(self.eps),
            ITERS=int(self.sinkhorn_iters),
            TPB=TPB,
            num_warps=1
        )
        return pre.view(b, s, 4), post.view(b, s, 4), comb.view(b, s, 4, 4)


def get_init_inputs():
    return [4, 20, 1e-6]


def get_inputs():
    hc = 4
    mix_hc = (2 + hc) * hc
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32, device="cuda")
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    hc_base = torch.zeros(mix_hc, dtype=torch.float32, device="cuda")
    return [mixes, hc_scale, hc_base]
