import torch, torch_npu
import triton
import triton.language as tl


@triton.jit
def fps_fused_kernel(
    x_ptr,            # (D, N) float32, 坐标主序
    out_ptr,          # (num_samples,) int32
    N,                # 点数 (runtime)
    num_samples,      # 采样数 (runtime, 作为 for 上界)
    start_idx,        # 起始点
    D: tl.constexpr,  # 维度 (静态展开)
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # 真实 lane = +inf (供 min 更新); padding lane = -inf (保证 argmax 不会选到)
    dist = tl.where(mask, float('inf'), float('-inf'))

    last = start_idx
    tl.store(out_ptr + 0, last)

    # 整个采样循环都在核内 —— 一次下发
    for i in range(1, num_samples):
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in tl.static_range(D):                      # D 小, 静态展开
            xd = tl.load(x_ptr + d * N + offs, mask=mask, other=0.0)
            pd = tl.load(x_ptr + d * N + last)            # 数据依赖标量 gather
            diff = xd - pd
            acc += diff * diff

        acc = tl.where(mask, acc, float('-inf'))
        dist = tl.minimum(dist, acc)                      # 更新全局最小距离
        last = tl.argmax(dist, axis=0).to(tl.int32)       # 选最远点
        tl.store(out_ptr + i, last)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, x, num_samples, random_start=True,
    ):
        N, D = x.shape
        device = x.device

        # 选择第一个点（种子点）
        if random_start:
            start_idx = int(torch.randint(0, N, (1,), device=device).item())
        else:
            start_idx = 0

        # 转坐标主序 (D, N) 连续, 便于核内逐维连续访存
        x_dn = x.t().contiguous()
        out = torch.empty(num_samples, dtype=torch.int32, device=device)

        BLOCK_N = triton.next_power_of_2(N)
        # 整个采样循环在一次 kernel 下发内完成
        fps_fused_kernel[(1,)](
            x_dn, out, N, num_samples, start_idx,
            D=D, BLOCK_N=BLOCK_N,
        )
        return out.to(torch.long)   # 与原版返回 long 一致


def get_inputs():
    x = torch.randn(1000, 3, device='npu')
    num_samples = 256
    return [x, num_samples]

def get_init_inputs():
    return []
