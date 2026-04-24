import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        x_f = x.float()
        residual_f = residual.float()
        post_f = post.float().unsqueeze(-1)
        comb_f = comb.float().unsqueeze(-1)
        output = post_f * x_f.unsqueeze(-2) + torch.sum(
            comb_f * residual_f.unsqueeze(-2), dim=2
        )
        return output.bfloat16()


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
    return Model(*get_init_inputs()).forward(*get_inputs())


def get_inputs():
    params = {'batch_size': 1, 'seq_len': 4096, 'hidden': 1280, 'hc': 4}
    x_data, residual_data, post_data, comb_data, o_grad = generate_test_data(params)
    return [x_data, residual_data, post_data, comb_data]


def get_init_inputs():
    return []
