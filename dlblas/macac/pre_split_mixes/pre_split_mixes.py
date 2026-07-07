import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Pure PyTorch implementation of mhc_pre_split_mixes.
    Applies per-channel scale + bias to input_mixes, then splits into:
      - pre_mix:  sigmoid(x[:mhc_mult])        + mhc_pre_eps  -> [*, mhc_mult, 1]
      - post_mix: sigmoid(x[mhc_mult:2*mhc_mult]) * mhc_post_mult_value -> [*, mhc_mult, 1]
      - comb_mix: x[2*mhc_mult:].view(*, mhc_mult, mhc_mult)
    """
    def __init__(
        self,
        mhc_mult: int,
        mhc_post_mult_value: float = 2.0,
        mhc_pre_eps: float = 1e-2,
    ):
        super().__init__()
        self.mhc_mult = mhc_mult
        self.mhc_post_mult_value = mhc_post_mult_value
        self.mhc_pre_eps = mhc_pre_eps
        mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
        self.mhc_scale = nn.Parameter(torch.randn(3) * 0.1)
        self.mhc_base = nn.Parameter(torch.randn(mhc_mult3) * 0.1)
    def forward(
        self,
        input_mixes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_mixes: [batch, seq_len, mhc_mult3] float32
        Returns:
            pre_mix:  [batch, seq_len, mhc_mult, 1]           float32
            post_mix: [batch, seq_len, mhc_mult, 1]           float32
            comb_mix: [batch, seq_len, mhc_mult, mhc_mult]    float32
        """
        m = self.mhc_mult
        a, b = input_mixes.shape[:2]
        scale = torch.cat([
            self.mhc_scale[0].expand(m),
            self.mhc_scale[1].expand(m),
            self.mhc_scale[2].expand(m * m),
        ])
        x = input_mixes * scale + self.mhc_base
        pre_mix = x[:, :, :m].sigmoid().unsqueeze(-1) + self.mhc_pre_eps
        post_mix = (x[:, :, m:2 * m].sigmoid() * self.mhc_post_mult_value).unsqueeze(-1)
        comb_mix = x[:, :, 2 * m:].view(a, b, m, m)
        return pre_mix, post_mix, comb_mix
n0 = 1
n1 = 1024
mhc_mult = 4
def get_inputs():
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    input_mixes = torch.randn(n0, n1, mhc_mult3)
    return [input_mixes]
def get_init_inputs():
    return [mhc_mult]