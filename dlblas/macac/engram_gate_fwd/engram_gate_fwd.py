import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
        """Pure PyTorch reference implementation of engram gate (vectorized, supports autograd).
        Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v
        Args:
            hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
            k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
            v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
            weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
            weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
            clamp_value: Clamp threshold for signed-sqrt gate activation.
            eps: Epsilon for RMSNorm numerical stability.
            save_for_backward: If True, also return (dot, gate_score, rstd_x, rstd_k).
        Returns:
            If save_for_backward is False: output tensor of shape (num_tokens, hc_mult, hidden_size), bfloat16.
            If save_for_backward is True: tuple of (output, dot, gate_score, rstd_x, rstd_k).
        """
        hidden_size = hidden_states.shape[-1]
        scalar = hidden_size**-0.5
        x = hidden_states.float()
        k_f = k.float()
        wh = weight_hidden.float().unsqueeze(0)
        we = weight_embed.float().unsqueeze(0)
        # RMSNorm
        rstd_x = torch.rsqrt(x.pow(2).mean(-1) + eps)
        rstd_k = torch.rsqrt(k_f.pow(2).mean(-1) + eps)
        # Dot -> sqrt-gate -> sigmoid
        # raw_dot is the unnormalized sum(x * wh * k * we), matching the kernel's dot_out
        raw_dot = torch.einsum('...d,...d->...', x * wh, k_f * we)
        dot = raw_dot * rstd_x * rstd_k * scalar
        signed_sqrt = dot.abs().clamp_min(clamp_value).sqrt() * dot.sign()
        gate_score = signed_sqrt.sigmoid()
        output = x + gate_score.unsqueeze(-1) * v.unsqueeze(-2)
        output = output.bfloat16()
        return output, raw_dot, gate_score, rstd_x, rstd_k
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
    return [ x_data, k_data, v_data, wh_data, we_data, clamp_value, eps]
def get_init_inputs():
    return []