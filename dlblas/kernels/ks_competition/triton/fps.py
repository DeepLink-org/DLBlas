import torch
import triton
import triton.language as tl


@triton.jit
def _update_min_distances_kernel(
    X_ptr, CP_ptr, Dist_ptr, N, D, stride_xn, stride_xd, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs < N

    # Accumulate squared distances to current point
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over feature dimension in fixed-size chunks using constexpr arange
    d0 = 0
    for i in range(0, tl.cdiv(D, 32)):
        # while d0 < D:
        cols = d0 + tl.arange(0, 32)  # constexpr chunk
        mask_d = cols < D

        # Load current point chunk [32]
        cp_vals = tl.load(CP_ptr + cols, mask=mask_d, other=0.0).to(tl.float32)

        # Load X tile [BLOCK_SIZE, 32]
        x_ptrs = X_ptr + offs[:, None] * stride_xn + cols[None, :] * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
            tl.float32
        )

        # Compute squared differences and reduce along D-chunk
        diff = x_vals - cp_vals[None, :]
        acc += tl.sum(diff * diff, axis=1)

        d0 += 32

    # Update distances: distances = min(distances, acc)
    old = tl.load(Dist_ptr + offs, mask=mask_n, other=0.0)
    new = tl.minimum(old, acc)
    tl.store(Dist_ptr + offs, new, mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

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
        # Precompute strides for Triton
        stride_xn, stride_xd = x.stride()
        BLOCK_SIZE = 256

        for i in range(1, num_samples):
            # 当前选中点
            current_point = x[
                selected[i - 1]
            ].contiguous()  # ensure contiguous for CP_ptr

            # 使用 Triton 内核更新最小距离
            grid = (triton.cdiv(N, BLOCK_SIZE),)
            _update_min_distances_kernel[grid](
                x,
                current_point,
                distances,
                N,
                D,
                stride_xn,
                stride_xd,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )

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
out = ModelNew(*get_init_inputs()).forward(*get_inputs())
print(out)
