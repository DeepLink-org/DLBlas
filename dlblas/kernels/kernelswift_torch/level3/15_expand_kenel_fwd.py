import torch
import torch.nn as nn

class Model(nn.Module):
    """
    算子：expand_to_mhc
    功能：在输入张量的倒数第二维插入一个新维度，并将其扩展 mhc_mult 倍。
    输入形状：(..., hidden_size)
    输出形状：(..., mhc_mult, hidden_size)
    """
    def __init__(self, mhc_mult: int):
        super(Model, self).__init__()
        self.mhc_mult = mhc_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行维度扩展操作。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 (batch, seq_len, hidden_dim)。

        Returns:
            torch.Tensor: 扩展后的张量，形状为 (batch, seq_len, mhc_mult, hidden_dim)。
        """
        original_shape = x.shape
        return x.unsqueeze(-2).expand(*original_shape[:-1], self.mhc_mult, original_shape[-1]).contiguous()


def get_init_inputs():
    """
    提供 Model 类初始化所需的参数。
    mhc_mult: 扩展倍数，参考原代码通常为 2, 4, 8 等。
    """
    return [4]  # 示例：mhc_mult = 4

def get_inputs():
    """
    提供 Model forward 函数所需的输入张量。
    参考原测试用例：n0=1, n1=1024, h=1280
    """
    # 创建一个形状为 (1, 1024, 1280) 的随机张量
    batch_size = 1
    seq_len = 1024
    hidden_dim = 1280
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return [x]