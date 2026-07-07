import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, clamp_value: float = 1e-6, eps: float = 1e-20):
        super().__init__()
        self.clamp_value = clamp_value
        self.eps = eps

    def forward(self, grad_out, x_data, k_data, v_data, wh_data, we_data):
        # Cast all inputs to float32
        x  = x_data.float()    # (T, H, D)
        k  = k_data.float()    # (T, H, D)
        v  = v_data.float()    # (T, D)
        wh = wh_data.float()   # (H, D)
        we = we_data.float()   # (H, D)
        go = grad_out.float()  # (T, H, D)

        D = x.shape[-1]
        scalar = D ** -0.5

        # ── Forward (recompute intermediates) ──────────────────────────
        rstd_x  = torch.rsqrt(x.pow(2).mean(-1) + self.eps)               # (T, H)
        rstd_k  = torch.rsqrt(k.pow(2).mean(-1) + self.eps)               # (T, H)
        raw_dot = (x * wh * (k * we)).sum(-1)                             # (T, H)
        dot     = raw_dot * rstd_x * rstd_k * scalar                      # (T, H)
        s_sqrt  = dot.abs().clamp_min(self.clamp_value).sqrt() * dot.sign()  # (T, H)
        gate    = s_sqrt.sigmoid()                                         # (T, H)

        # ── Backward ───────────────────────────────────────────────────
        # output = x + gate[T,H,1] * v[T,1,D]

        # grad_v: Σ_h go[t,h,d] * gate[t,h]
        grad_v = (go * gate.unsqueeze(-1)).sum(1)                          # (T, D)

        # grad_gate: Σ_d go[t,h,d] * v[t,d]
        grad_gate = (go * v.unsqueeze(1)).sum(-1)                          # (T, H)

        # grad through sigmoid: d(sigmoid)/d(s) = gate*(1-gate)
        grad_s_sqrt = grad_gate * gate * (1.0 - gate)                     # (T, H)

        # grad_dot: d(sign(dot)*√|dot|.clamp(cv))/d(dot) = 0.5/√|dot| when |dot|≥cv, else 0
        mask    = (dot.abs() >= self.clamp_value).float()
        grad_dot = grad_s_sqrt * mask * 0.5 / dot.abs().clamp(min=self.clamp_value).sqrt()  # (T, H)

        # grad from dot = raw_dot * rstd_x * rstd_k * scalar
        grad_raw_dot = grad_dot * rstd_x * rstd_k * scalar                # (T, H)
        grad_rstd_x  = grad_dot * raw_dot * rstd_k * scalar               # (T, H)
        grad_rstd_k  = grad_dot * raw_dot * rstd_x * scalar               # (T, H)

        # grad_x  (三条路径)
        #   path A (direct):   go
        #   path B (raw_dot):  grad_raw_dot * wh * (k * we)
        #   path C (rstd_x):   grad_rstd_x * (-x/D) * rstd_x³
        grad_x = (go
                  + grad_raw_dot.unsqueeze(-1) * wh * (k * we)
                  + grad_rstd_x.unsqueeze(-1) * (-x / D) * rstd_x.unsqueeze(-1).pow(3))  # (T, H, D)

        # grad_k  (两条路径)
        #   path A (raw_dot):  grad_raw_dot * we * (x * wh)
        #   path B (rstd_k):   grad_rstd_k * (-k/D) * rstd_k³
        grad_k = (grad_raw_dot.unsqueeze(-1) * we * (x * wh)
                  + grad_rstd_k.unsqueeze(-1) * (-k / D) * rstd_k.unsqueeze(-1).pow(3))  # (T, H, D)

        # grad_wh, grad_we  (raw_dot[t,h] = Σ_d x*wh*k*we，对 T 求和)
        grad_wh = (grad_raw_dot.unsqueeze(-1) * (k * we) * x).sum(0)      # (H, D)
        grad_we = (grad_raw_dot.unsqueeze(-1) * (x * wh) * k).sum(0)      # (H, D)

        return grad_x, grad_k, grad_v, grad_wh, grad_we


num_tokens = 14
hc_mult = 4
hidden_size = 128
eps = 1e-20
clamp_value = 1e-6


def generate_test_data(num_tokens, hc_mult, hidden_size):
    x_data    = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    k_data    = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    v_data    = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    wh_data   = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
    we_data   = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
    grad_out  = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    return [grad_out, x_data, k_data, v_data, wh_data, we_data]


def get_inputs():
    return generate_test_data(num_tokens, hc_mult, hidden_size)


def get_init_inputs():
    return [clamp_value, eps]