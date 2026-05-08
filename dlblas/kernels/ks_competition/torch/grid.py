import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, pos, size, start=None, end=None):
        if pos.dim() != 2:
            raise ValueError(
                f"pos should be 2-dimensional, got {pos.dim()}-dimensional"
            )

        if size.dim() != 1:
            raise ValueError(
                f"size should be 1-dimensional, got {size.dim()}-dimensional"
            )

        if pos.size(1) != size.size(0):
            raise ValueError(
                f"Dimension mismatch: pos has {pos.size(1)} dimensions, "
                f"but size has {size.size(0)} dimensions"
            )

        N, D = pos.shape
        device = pos.device

        # 处理可选参数
        if start is None:
            start = torch.zeros(D, device=device)
        else:
            if start.dim() != 1 or start.size(0) != D:
                raise ValueError(f"start should have shape [{D}], got {start.shape}")

        if end is None:
            # 如果没有提供end，则使用点的最大坐标
            end = torch.max(pos, dim=0)[0] + size
        else:
            if end.dim() != 1 or end.size(0) != D:
                raise ValueError(f"end should have shape [{D}], got {end.shape}")

        # 将点坐标转换为网格索引
        grid_indices = ((pos - start.unsqueeze(0)) / size.unsqueeze(0)).long()

        # 确保网格索引在有效范围内
        grid_indices = torch.clamp(grid_indices, min=0)

        # 计算每个维度上的网格数量
        grid_counts = ((end - start) / size).long() + 1

        # 计算每个点的唯一网格ID
        cluster_ids = torch.zeros(N, dtype=torch.long, device=device)

        # 使用多维网格索引计算唯一ID
        for d in range(D):
            if d == 0:
                cluster_ids = grid_indices[:, d]
            else:
                cluster_ids = cluster_ids * grid_counts[d] + grid_indices[:, d]

        # 重新映射聚类ID为连续的整数
        unique_ids, inverse_indices = torch.unique(cluster_ids, return_inverse=True)

        return inverse_indices


def get_inputs():
    pos = torch.tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]], device="npu")
    size = torch.tensor([5, 5], device="npu")
    end = torch.tensor([19, 19], device="npu")
    return [pos, size, end]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = Model(*get_init_inputs()).forward(*get_inputs())
print(out)
