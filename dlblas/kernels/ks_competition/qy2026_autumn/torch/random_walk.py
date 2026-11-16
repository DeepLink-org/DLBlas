import torch
import random


def _build_adjacency_list(row: torch.Tensor, col: torch.Tensor, num_nodes: int) -> list:
    adj_list = [[] for _ in range(num_nodes)]

    for edge_idx in range(row.size(0)):
        src = row[edge_idx].item()
        dst = col[edge_idx].item()

        # 确保节点索引在有效范围内
        if src < num_nodes and dst < num_nodes:
            adj_list[src].append((dst, edge_idx))

    return adj_list


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, row, col, start, walk_length, num_nodes=None, return_edge_indices=False
    ):
        if row.dim() != 1 or col.dim() != 1:
            raise ValueError("row and col should be 1-dimensional tensors")

        if row.size(0) != col.size(0):
            raise ValueError("row and col should have the same length")

        if start.dim() != 1:
            raise ValueError("start should be 1-dimensional tensor")

        if walk_length <= 0:
            raise ValueError("walk_length should be positive")

        device = row.device
        num_starts = start.size(0)

        # 确定节点总数
        if num_nodes is None:
            num_nodes = max(row.max().item(), col.max().item(), start.max().item()) + 1

        # 构建邻接表
        adj_list = _build_adjacency_list(row, col, num_nodes)

        # 初始化节点序列
        node_seq = torch.zeros(
            num_starts, walk_length + 1, dtype=torch.long, device=device
        )
        node_seq[:, 0] = start

        # 初始化边索引（如果需要返回）
        if return_edge_indices:
            edge_seq = torch.full(
                (num_starts, walk_length), -1, dtype=torch.long, device=device
            )

        # 为每个起始节点执行随机游走
        for i in range(num_starts):
            current_node = start[i].item()

            for step in range(1, walk_length + 1):
                # 获取当前节点的邻居
                neighbors = adj_list[current_node]

                if len(neighbors) == 0:
                    # 没有邻居，停留在当前节点
                    next_node = current_node
                    edge_idx = -1
                else:
                    # 随机选择一个邻居
                    neighbor_idx = random.randint(0, len(neighbors) - 1)
                    next_node, edge_idx = neighbors[neighbor_idx]

                # 更新序列
                node_seq[i, step] = next_node
                if return_edge_indices:
                    edge_seq[i, step - 1] = edge_idx

                current_node = next_node

        if return_edge_indices:
            return node_seq, edge_seq
        else:
            return node_seq


def get_inputs():
    num_nodes = 5
    walk_length = 8
    row = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device="npu")
    col = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long, device="npu")
    start = torch.arange(num_nodes, dtype=torch.long, device="npu")
    return [row, col, start, walk_length]


def get_init_inputs():
    return []


out = Model(*get_init_inputs()).forward(*get_inputs())
