import torch


def _knn_single_batch(x, y, k, cosine=False):
    """
    单批次的knn计算（内部函数）
    """
    N, F = x.shape
    M, _ = y.shape

    if cosine:
        # 使用余弦相似度
        # 归一化向量
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)

        # 计算余弦相似度（越大越相似）
        similarity = torch.matmul(x_norm, y_norm.t())

        # 获取top-k相似度的索引（largest=True）
        topk_values, topk_indices = torch.topk(
            similarity, k=min(k, M), dim=1, largest=True
        )

    else:
        # 使用欧氏距离
        # 计算距离矩阵: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        x_squared = (x**2).sum(dim=1, keepdim=True)  # [N, 1]
        y_squared = (y**2).sum(dim=1, keepdim=True)  # [M, 1]

        # 计算点积
        xy = torch.matmul(x, y.t())  # [N, M]

        # 计算距离矩阵
        distance = x_squared + y_squared.t() - 2 * xy  # [N, M]

        # 数值稳定性
        distance = torch.clamp(distance, min=0)

        # 获取top-k最小距离的索引
        topk_values, topk_indices = torch.topk(
            distance, k=min(k, M), dim=1, largest=False
        )

    # 构建row和col索引
    row = torch.arange(N, device=x.device).repeat_interleave(k)  # [N*k]
    col = topk_indices.flatten()  # [N*k]

    return row, col


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x,
        y,
        k,
        batch_x=None,
        batch_y=None,
        cosine=False,
    ):
        N, F = x.shape
        M, _ = y.shape

        # 处理批次情况
        if batch_x is not None and batch_y is not None:
            # 按批次分别处理
            unique_batches = torch.unique(batch_x)
            rows = []
            cols = []

            for batch_id in unique_batches:
                # 获取当前批次的索引
                x_mask = batch_x == batch_id
                y_mask = batch_y == batch_id

                x_batch = x[x_mask]
                y_batch = y[y_mask]

                # 获取批次内的索引
                x_indices = torch.nonzero(x_mask).squeeze(1)
                y_indices = torch.nonzero(y_mask).squeeze(1)

                # 计算批次内的knn
                batch_row, batch_col = _knn_single_batch(x_batch, y_batch, k, cosine)

                # 转换为全局索引
                rows.append(x_indices[batch_row])
                cols.append(y_indices[batch_col])

            return torch.cat(rows), torch.cat(cols)

        else:
            # 无批次情况
            return _knn_single_batch(x, y, k, cosine)


def get_inputs():
    x = torch.randn(15, 3, device="npu")
    y = torch.randn(25, 3, device="npu")
    k = 2
    batch_x = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], device="npu")
    batch_y = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        device="npu",
    )
    return [x, y, k, batch_x, batch_y]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = Model(*get_init_inputs()).forward(*get_inputs())
print(out)
