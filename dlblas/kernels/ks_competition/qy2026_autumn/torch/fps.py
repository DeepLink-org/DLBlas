import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x,
        num_samples,
        random_start=True,
    ):
        N, D = x.shape
        device = x.device

        # 初始化
        distances = torch.full((N,), float("inf"), device=device)
        selected = torch.zeros(num_samples, dtype=torch.long, device=device)

        # 选择第一个点
        if random_start:
            start_idx = torch.randint(0, N, (1,), device=device)
        else:
            start_idx = torch.tensor([0], device=device)

        selected[0] = start_idx
        distances[start_idx] = 0

        # 迭代选择剩余点
        for i in range(1, num_samples):
            # 计算当前选中点到所有点的距离
            current_point = x[selected[i - 1]].unsqueeze(0)
            dist_to_current = torch.sum((x - current_point) ** 2, dim=1)

            # 更新最小距离
            distances = torch.min(distances, dist_to_current)

            # 选择距离最大的点
            selected[i] = torch.argmax(distances)

        return selected


def get_inputs():
    x = torch.randn(1000, 3, device="npu")
    num_samples = 256
    return [x, num_samples]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = Model(*get_init_inputs()).forward(*get_inputs())
print(out)
