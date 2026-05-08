import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, batch_x=None, batch_y=None):
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("x and y should be 2-dimensional tensors")

        if x.size(1) != y.size(1):
            raise ValueError(
                f"x and y should have the same feature dimension, "
                f"got {x.size(1)} and {y.size(1)}"
            )

        device = x.device
        N, D = x.shape
        M = y.shape[0]

        # 处理批次参数
        if batch_x is not None and batch_y is not None:
            # 验证批次索引
            _validate_batch_indices(batch_x, batch_y, N, M)

            # 按批次处理
            return _batch_nearest(x, y, batch_x, batch_y)
        elif batch_x is None and batch_y is None:
            # 单批次处理
            return _single_batch_nearest(x, y)
        else:
            raise ValueError("batch_x and batch_y should be both provided or both None")


def _validate_batch_indices(
    batch_x: torch.Tensor, batch_y: torch.Tensor, N: int, M: int
):
    """验证批次索引的正确性"""
    if batch_x.dim() != 1 or batch_y.dim() != 1:
        raise ValueError("batch_x and batch_y should be 1-dimensional tensors")

    if batch_x.size(0) != N:
        raise ValueError(f"batch_x size {batch_x.size(0)} should match x size {N}")

    if batch_y.size(0) != M:
        raise ValueError(f"batch_y size {batch_y.size(0)} should match y size {M}")

    # 检查批次索引是否有序
    if not _is_sorted(batch_x):
        raise ValueError("batch_x should be sorted")

    if not _is_sorted(batch_y):
        raise ValueError("batch_y should be sorted")

    # 检查批次是否匹配
    unique_x = torch.unique(batch_x)
    unique_y = torch.unique(batch_y)

    if not torch.equal(unique_x, unique_y):
        raise ValueError(
            "batch_x and batch_y should have the same unique batch indices"
        )


def _is_sorted(tensor: torch.Tensor) -> bool:
    """检查张量是否有序"""
    if tensor.numel() == 0:
        return True

    # 检查是否非递减
    diff = tensor[1:] - tensor[:-1]
    return (diff >= 0).all()


def _single_batch_nearest(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """单批次最近邻查找"""
    # 计算所有点对之间的距离
    # x: [N, D], y: [M, D] -> dist: [N, M]
    dist = torch.cdist(x, y, p=2)

    # 找到每个x点的最近y点索引
    min_dist, indices = torch.min(dist, dim=1)

    return indices


def _batch_nearest(
    x: torch.Tensor, y: torch.Tensor, batch_x: torch.Tensor, batch_y: torch.Tensor
) -> torch.Tensor:
    """批量最近邻查找"""
    device = x.device
    N = x.size(0)

    # 获取唯一的批次索引
    unique_batches = torch.unique(batch_x)

    # 为每个批次单独处理
    all_indices = torch.zeros(N, dtype=torch.long, device=device)

    for batch_id in unique_batches:
        # 获取当前批次的x和y点
        x_mask = batch_x == batch_id
        y_mask = batch_y == batch_id

        batch_x_points = x[x_mask]
        batch_y_points = y[y_mask]

        if len(batch_y_points) == 0:
            raise ValueError(f"Batch {batch_id} has no points in y")

        # 计算当前批次的最近邻
        batch_indices = _single_batch_nearest(batch_x_points, batch_y_points)

        # 将批次内的索引转换为全局索引
        y_global_indices = torch.where(y_mask)[0]
        global_indices = y_global_indices[batch_indices]

        # 存储结果
        all_indices[x_mask] = global_indices

    return all_indices


def get_inputs():
    x = torch.tensor(
        [
            [-1, -1],
            [-1, +1],
            [+1, +1],
            [+1, -1],
            [-2, -2],
            [-2, +2],
            [+2, +2],
            [+2, -2],
        ],
        dtype=torch.float,
        device="npu",
    )

    y = torch.tensor(
        [
            [-1, 0],
            [+1, 0],
            [-2, 0],
            [+2, 0],
        ],
        dtype=torch.float,
        device="npu",
    )

    batch_x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device="npu")
    batch_y = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="npu")
    return [x, y, batch_x, batch_y]


def get_init_inputs():
    return []


out = Model(*get_init_inputs()).forward(*get_inputs())
