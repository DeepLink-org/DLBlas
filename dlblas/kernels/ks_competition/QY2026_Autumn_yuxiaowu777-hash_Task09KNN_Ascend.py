"""
KNN — NPU 优化版本

优化策略:
  1. 消除 Python 批次循环: 计算全量距离矩阵, 跨批次置 inf, 单次 topk
     原版 unique → mask → per-batch compute 变为一次矩阵运算
  2. 距离矩阵用 mm 计算 (||x-y||^2 = ||x||^2+||y||^2-2x·y), Cube Core 加速
  3. 批次掩码用广播比较一步生成, 无需循环
"""

import torch
import torch.nn as nn
import torch_npu


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, y, k, batch_x=None, batch_y=None, cosine=False):
        N, F = x.shape
        M, _ = y.shape
        actual_k = min(k, M)

        if cosine:
            x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
            y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)
            scores = torch.mm(x_norm, y_norm.t())
            if batch_x is not None and batch_y is not None:
                cross_batch = batch_x.unsqueeze(1) != batch_y.unsqueeze(0)
                scores.masked_fill_(cross_batch, -float('inf'))
            _, topk_idx = torch.topk(scores, k=actual_k, dim=1, largest=True)
        else:
            x_sq = (x * x).sum(dim=1, keepdim=True)
            y_sq = (y * y).sum(dim=1, keepdim=True)
            dist = x_sq + y_sq.t() - 2.0 * torch.mm(x, y.t())
            dist.clamp_(min=0.0)
            if batch_x is not None and batch_y is not None:
                cross_batch = batch_x.unsqueeze(1) != batch_y.unsqueeze(0)
                dist.masked_fill_(cross_batch, float('inf'))
            _, topk_idx = torch.topk(dist, k=actual_k, dim=1, largest=False)

        row = torch.arange(N, device=x.device).repeat_interleave(k)
        col = topk_idx.reshape(-1)
        return row, col


def get_inputs():
    device = 'npu'
    x = torch.randn(15, 3, device=device)
    y = torch.randn(25, 3, device=device)
    k = 2
    batch_x = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2], device=device)
    batch_y = torch.tensor([0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,2,2,2], device=device)
    return [x, y, k, batch_x, batch_y]


def get_init_inputs():
    return []


if __name__ == "__main__":
    torch.set_default_device("npu")
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    row, col = model(*inputs)
    print(f"row: {row}")
    print(f"col: {col}")
    print(f"Shape: row={row.shape}, col={col.shape}")
