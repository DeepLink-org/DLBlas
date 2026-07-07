import torch
import torch.nn as nn


class Model(nn.Module):
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

        x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
        base = hc_base.to(dtype=torch.float32)
        s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

        pre = torch.sigmoid(x[:, :hc] * s0 + base[:hc].unsqueeze(0)) + eps
        post = 2 * torch.sigmoid(x[:, hc : 2 * hc] * s1 + base[hc : 2 * hc].unsqueeze(0))
        raw = x[:, 2 * hc : 2 * hc + hc * hc]
        comb = raw.view(-1, hc, hc) * s2 + base[2 * hc : 2 * hc + hc * hc].view(1, hc, hc)

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