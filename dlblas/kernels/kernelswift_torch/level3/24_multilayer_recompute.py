import torch
import torch.nn as nn


class Model(nn.Module):
    """
    算子：multilayer_recompute
    功能：逐层计算 layer_input，并在需要时更新 residual。
    输入：
        initial_residual: [batch, seq, mhc_mult, hidden]
        pre_mix_list: list[[batch, seq, mhc_mult, 1]]
        layer_output_list: list[[batch, seq, hidden]]
        post_mix_list: list[[batch, seq, mhc_mult, 1]]
        comb_mix_list: list[[batch, seq, mhc_mult, mhc_mult]]
    输出：
        layer_input_list: list[[batch, seq, hidden]]
        residual_list: list[[batch, seq, mhc_mult, hidden]]
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        initial_residual: torch.Tensor,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
    ):
        layer_input_list = []
        residual_list = []

        residual = initial_residual
        for i in range(len(pre_mix_list)):
            layer_input = (residual * pre_mix_list[i]).sum(dim=2).to(torch.bfloat16)
            layer_input_list.append(layer_input)

            if i < len(layer_output_list):
                residual_fp32 = residual.float()
                comb = comb_mix_list[i].transpose(-1, -2).float()
                term2 = torch.matmul(comb, residual_fp32)
                residual = (
                    layer_output_list[i].float().unsqueeze(2) * post_mix_list[i]
                    + term2
                ).to(torch.bfloat16)
                residual_list.append(residual)

        return layer_input_list, residual_list


batch_size = 1
seq_len = 8192
mhc_mult = 4
hidden = 2560
num_layers = 2
num_post = 1


def get_init_inputs():
    return []


def get_inputs():
    initial_residual = torch.randn(
        batch_size, seq_len, mhc_mult, hidden, dtype=torch.bfloat16
    )
    pre_mix_list = [
        torch.randn(batch_size, seq_len, mhc_mult, 1, dtype=torch.float32)
        for _ in range(num_layers)
    ]
    layer_output_list = [
        torch.randn(batch_size, seq_len, hidden, dtype=torch.bfloat16)
        for _ in range(num_post)
    ]
    post_mix_list = [
        torch.randn(batch_size, seq_len, mhc_mult, 1, dtype=torch.float32)
        for _ in range(num_post)
    ]
    comb_mix_list = [
        torch.randn(batch_size, seq_len, mhc_mult, mhc_mult, dtype=torch.float32)
        for _ in range(num_post)
    ]
    return [initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list]
