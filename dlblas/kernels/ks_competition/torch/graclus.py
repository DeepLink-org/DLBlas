import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, row, col, weight=None):
        if row.dim() != 1 or col.dim() != 1:
            raise ValueError("row and col should be 1-dimensional tensors")

        if row.size(0) != col.size(0):
            raise ValueError("row and col should have the same length")

        if weight is not None and weight.dim() != 1:
            raise ValueError("weight should be 1-dimensional tensor")

        if weight is not None and weight.size(0) != row.size(0):
            raise ValueError("weight should have the same length as row and col")

        device = row.device
        num_edges = row.size(0)

        # 确定节点数量
        num_nodes = max(row.max().item(), col.max().item()) + 1

        # 如果没有提供权重，使用默认权重1.0
        if weight is None:
            weight = torch.ones(num_edges, device=device)

        # 构建邻接表
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(num_edges):
            src, dst, w = row[i].item(), col[i].item(), weight[i].item()
            adj_list[src].append((dst, w))
            adj_list[dst].append((src, w))  # 无向图

        # 计算节点权重（使用度或加权度）
        node_weights = torch.zeros(num_nodes, device=device)
        for i in range(num_nodes):
            if weight is not None:
                # 使用加权度作为节点权重
                node_weights[i] = sum(w for _, w in adj_list[i])
            else:
                # 使用度作为节点权重
                node_weights[i] = len(adj_list[i])

        # 贪心匹配算法
        matching = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # 按权重降序排序节点
        sorted_indices = torch.argsort(node_weights, descending=True)

        for node in sorted_indices:
            node = node.item()
            if visited[node]:
                continue

            visited[node] = True

            # 获取未访问的邻居
            unvisited_neighbors = []
            for neighbor, w in adj_list[node]:
                if not visited[neighbor]:
                    unvisited_neighbors.append((neighbor, w))

            if len(unvisited_neighbors) == 0:
                # 没有未访问的邻居，保持未匹配状态
                continue

            # 选择权重最大的未访问邻居进行匹配
            unvisited_neighbors.sort(key=lambda x: x[1], reverse=True)
            best_neighbor = unvisited_neighbors[0][0]

            # 建立匹配
            matching[node] = best_neighbor
            matching[best_neighbor] = node
            visited[best_neighbor] = True

        # 构建聚类结果
        cluster = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        cluster_id = 0

        for i in range(num_nodes):
            if cluster[i] != -1:
                continue

            if matching[i] == -1:
                # 单个节点构成一个聚类
                cluster[i] = cluster_id
                cluster_id += 1
            else:
                # 匹配的节点对构成一个聚类
                j = matching[i]
                cluster[i] = cluster_id
                cluster[j] = cluster_id
                cluster_id += 1

        return cluster


def get_inputs():
    row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], device="npu")
    col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 3, 1, 2], device="npu")
    return [row, col]


def get_init_inputs():
    return []


out = Model(*get_init_inputs()).forward(*get_inputs())
