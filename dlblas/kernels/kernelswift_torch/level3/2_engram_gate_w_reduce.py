import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed):
        grad_w_sum = grad_w_partial.sum(0)
        grad_weight_hidden += grad_w_sum * weight_embed.float()
        grad_weight_embed += grad_w_sum * weight_hidden.float()
        return grad_weight_hidden, grad_weight_embed
def generate_test_data(hidden_size):
    hc_mult = 4
    num_persistent_blocks = 108
    grad_w_partial = torch.randn(num_persistent_blocks, hc_mult, hidden_size, dtype=torch.float32)
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    return (grad_w_partial, weight_hidden, weight_embed)
def test_engram_grad_w_reduce():    
    return Model(*get_init_inputs()).forward(*get_inputs())
def get_inputs():
    hidden_size = 4096
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(hidden_size)
    hc_mult = grad_w_partial.shape[1]
    grad_wh_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cpu')
    grad_we_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cpu')
    return [grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref]
def get_init_inputs():
    return []