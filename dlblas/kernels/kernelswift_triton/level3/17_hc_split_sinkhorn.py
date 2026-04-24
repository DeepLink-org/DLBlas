import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@triton.jit
def _modelnew_kernel(
    x_ptr,          # pointer to [N, MIX_HC]
    base_ptr,       # pointer to [MIX_HC]
    pre_ptr,        # pointer to [N, HC]
    post_ptr,       # pointer to [N, HC]
    comb_ptr,       # pointer to [N, HC*HC]
    N,              # total rows = b*s
    MIX_HC,         # cols in x
    s0, s1, s2,     # scaling factors (float32)
    eps,            # epsilon (float32)
    HC: tl.constexpr,                # head count (compile-time constant)
    SINKHORN_ITERS: tl.constexpr,    # iterations (compile-time constant)
):
    pid = tl.program_id(0)
    # Guard in case grid overlaunches
    if pid >= N:
        return

    # Base offsets
    x_row_ptr = x_ptr + pid * MIX_HC

    # Offsets for segments
    idx_h = tl.arange(0, HC)
    mask_h = idx_h < HC

    # pre = sigmoid(x[:hc] * s0 + base[:hc]) + eps
    x_pre = tl.load(x_row_ptr + idx_h, mask=mask_h, other=0.0)
    base_pre = tl.load(base_ptr + idx_h, mask=mask_h, other=0.0)
    z_pre = x_pre * s0 + base_pre
    pre = tl.sigmoid(z_pre) + eps
    tl.store(pre_ptr + pid * HC + idx_h, pre, mask=mask_h)

    # post = 2 * sigmoid(x[hc:2hc] * s1 + base[hc:2hc])
    x_post = tl.load(x_row_ptr + HC + idx_h, mask=mask_h, other=0.0)
    base_post = tl.load(base_ptr + HC + idx_h, mask=mask_h, other=0.0)
    z_post = x_post * s1 + base_post
    post = 2.0 * tl.sigmoid(z_post)
    tl.store(post_ptr + pid * HC + idx_h, post, mask=mask_h)

    # comb = raw.view(HC,HC) * s2 + base_mat.view(HC,HC)
    rows = tl.arange(0, HC)[:, None]
    cols = tl.arange(0, HC)[None, :]
    mat_off = rows * HC + cols
    mask_mat = mat_off < (HC * HC)

    x_mat = tl.load(x_row_ptr + 2 * HC + mat_off, mask=mask_mat, other=0.0)
    base_mat = tl.load(base_ptr + 2 * HC + mat_off, mask=mask_mat, other=0.0)
    mat = x_mat * s2 + base_mat  # shape (HC, HC)

    # Stabilize with row-wise max, then exponentiate
    row_max = tl.max(mat, axis=1)  # shape: (HC,)
    mat = tl.exp(mat - row_max[:, None])

    # First row normalization (no eps in denom) then +eps to matrix
    row_sum = tl.sum(mat, axis=1)  # shape: (HC,)
    mat = mat / row_sum[:, None] + eps

    # First column normalization with +eps in denom
    col_sum = tl.sum(mat, axis=0)
    mat = mat / (col_sum[None, :] + eps)

    # Remaining Sinkhorn iterations
    if SINKHORN_ITERS > 1:
        for _ in tl.static_range(SINKHORN_ITERS - 1):
            # Row normalization with +eps in denom
            row_sum = tl.sum(mat, axis=1)
            mat = mat / (row_sum[:, None] + eps)
            # Column normalization with +eps in denom
            col_sum = tl.sum(mat, axis=0)
            mat = mat / (col_sum[None, :] + eps)

    # Store comb flattened
    tl.store(comb_ptr + pid * (HC * HC) + mat_off, mat, mask=mask_mat)


class ModelNew(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, mix_hc = mixes.shape
        hc = self.hc_mult
        eps = self.eps
        expected = (2 + hc) * hc
        if mix_hc != expected:
            raise ValueError(f"expected mix dim {expected}, got {mix_hc}")

        # Common preprocessing
        x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
        base = hc_base.to(dtype=torch.float32)
        s0_t, s1_t, s2_t = hc_scale[0], hc_scale[1], hc_scale[2]

        # If Triton available and tensors on CUDA, run fused kernel
        if _TRITON_AVAILABLE and x.is_cuda:
            N = x.shape[0]
            # Ensure contiguity and device consistency
            x = x.contiguous()
            base = base.contiguous().to(x.device)

            pre_out = torch.empty((N, hc), dtype=torch.float32, device=x.device)
            post_out = torch.empty((N, hc), dtype=torch.float32, device=x.device)
            comb_out = torch.empty((N, hc * hc), dtype=torch.float32, device=x.device)

            # Convert scales to Python floats to pass as scalars
            s0 = float(s0_t.item() if isinstance(s0_t, torch.Tensor) else s0_t)
            s1 = float(s1_t.item() if isinstance(s1_t, torch.Tensor) else s1_t)
            s2 = float(s2_t.item() if isinstance(s2_t, torch.Tensor) else s2_t)

            grid = lambda META: (N,)
            _modelnew_kernel[grid](
                x, base, pre_out, post_out, comb_out,
                N, mix_hc,
                s0, s1, s2, float(eps),
                HC=hc,
                SINKHORN_ITERS=self.sinkhorn_iters,
                num_warps=1,
                num_stages=1,
            )

            return pre_out.view(b, s, hc), post_out.view(b, s, hc), comb_out.view(b, s, hc, hc)

        # Fallback PyTorch path (CPU or if Triton unavailable)
        pre = torch.sigmoid(x[:, :hc] * s0_t + base[:hc].unsqueeze(0)) + eps
        post = 2 * torch.sigmoid(x[:, hc : 2 * hc] * s1_t + base[hc : 2 * hc].unsqueeze(0))
        raw = x[:, 2 * hc : 2 * hc + hc * hc]
        comb = raw.view(-1, hc, hc) * s2_t + base[2 * hc : 2 * hc + hc * hc].view(1, hc, hc)

        row_max = comb.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb - row_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        return pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc)


def get_init_inputs():
    """Returns positional args for Model.__init__: (hc_mult, sinkhorn_iters, eps)."""
    return [4, 20, 1e-6]


def get_inputs():
    """Returns positional args for Model.forward: (mixes, hc_scale, hc_base)."""
    hc = 4
    mix_hc = (2 + hc) * hc
    torch.manual_seed(0)
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32)
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32)
    hc_base = torch.randn(mix_hc, dtype=torch.float32) * 0.1
    return [mixes, hc_scale, hc_base]
