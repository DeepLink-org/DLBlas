"""
FPS (Farthest Point Sampling) — NPU 优化版本

优化策略:
  1. 预计算 squared distance matrix 用 mm (NPU Cube Core 加速)
     ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b，一次矩阵乘完成
  2. 将 distance matrix 搬回 CPU 做 256 步贪心循环
     N=1000 小规模下 CPU 逐行 min+argmax 远快于 256 次 NPU kernel launch
  3. 结果搬回 NPU 返回
"""

import torch
import torch.nn as nn
import torch_npu


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, num_samples, random_start=True):
        N, D = x.shape
        device = x.device

        x_sq = (x * x).sum(dim=1)
        dist_matrix = x_sq.unsqueeze(1) + x_sq.unsqueeze(0) - 2.0 * torch.mm(x, x.t())
        dist_matrix.clamp_(min=0.0)

        dm = dist_matrix.cpu()

        distances = torch.full((N,), float('inf'), device='cpu')
        selected = torch.zeros(num_samples, dtype=torch.long, device='cpu')

        if random_start:
            idx = torch.randint(0, N, (1,), device=device).item()
        else:
            idx = 0

        selected[0] = idx

        for i in range(1, num_samples):
            dist_to_current = dm[idx]
            torch.minimum(distances, dist_to_current, out=distances)
            idx = torch.argmax(distances).item()
            selected[i] = idx

        return selected.to(device)


def get_inputs():
    x = torch.randn(1000, 3, device='npu')
    num_samples = 256
    return [x, num_samples]


def get_init_inputs():
    return []


if __name__ == "__main__":
    torch.set_default_device("npu")
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    print(output)
    print(f"Shape: {output.shape}")
