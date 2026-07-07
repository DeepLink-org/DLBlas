import torch
import torch.nn as nn


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


class Model(nn.Module):
    """
    Pure PyTorch implementation of the MHC pre-processing fused kernel.

    Pipeline:
      1. RMS-normalized linear projection of residual (mhc_pre_norm_fn)
      2. Split mixing logits into pre / post / comb components (mhc_pre_split_mixes)
      3. Sinkhorn doubly-stochastic normalization of comb_mix
      4. Weighted sum of MHC heads with pre_mix to produce layer_input
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