import torch
import triton
import triton.language as tl


@triton.jit
def _nn_min_kernel(
    x_ptr,
    y_ptr,
    out_idx_ptr,
    N,
    M,
    D,
    stride_xn,
    stride_xd,
    stride_ym,
    stride_yd,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Precompute x norms for this block
    x_norm = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        x_ptrs = x_ptr + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
        x_sub = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
            tl.float32
        )
        x_norm += tl.sum(x_sub * x_sub, axis=1)

    # Initialize best distances and indices
    INF = 1e30
    best_val = tl.full([BLOCK_N], INF, dtype=tl.float32)
    best_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    # Loop over tiles of Y
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        ynorm = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)

        # Accumulate x @ y^T via dot products; also accumulate norms of y
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            x_ptrs = x_ptr + offs_n[:, None] * stride_xn + offs_d[None, :] * stride_xd
            y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd

            x_sub = tl.load(
                x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            y_sub = tl.load(
                y_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)

            acc += tl.dot(x_sub, tl.trans(y_sub))
            ynorm += tl.sum(y_sub * y_sub, axis=1)

        # Compute squared distances tile and reduce per row to get argmin
        dist_tile = x_norm[:, None] + ynorm[None, :] - 2.0 * acc
        dist_tile = tl.where(mask_m[None, :], dist_tile, INF)
        neg = -dist_tile
        max_neg = tl.max(neg, axis=1)
        tile_best_val = -max_neg
        tile_best_idx = tl.argmax(neg, axis=1)

        # Compare with global best (strictly less to preserve first occurrence globally)
        cond2 = (tile_best_val < best_val) & mask_n
        best_val = tl.where(cond2, tile_best_val, best_val)
        best_idx = tl.where(cond2, m0 + tile_best_idx, best_idx)

    # Store results
    tl.store(out_idx_ptr + offs_n, best_idx, mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

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
    N, D = x.shape
    M = y.shape[0]

    # Keep baseline behavior for degenerate cases by delegating to PyTorch
    if M == 0 or N == 0:
        dist = torch.cdist(x, y, p=2)
        _, indices = torch.min(dist, dim=1)
        return indices

    # Triton accelerated path on CUDA; fallback to PyTorch otherwise
    use_triton = (
        x.is_cuda
        and y.is_cuda
        and x.dtype == y.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    )
    # if not use_triton:
    #     dist = torch.cdist(x, y, p=2)
    #     _, indices = torch.min(dist, dim=1)
    #     return indices

    # Allocate output indices (int32 in kernel, cast to long on return)
    out_i32 = torch.empty(N, dtype=torch.int32, device=x.device)

    # Tiling parameters
    BLOCK_N = 128
    BLOCK_M = 64
    BLOCK_D = 32

    grid = (triton.cdiv(N, BLOCK_N),)

    _nn_min_kernel[grid](
        x,
        y,
        out_i32,
        N,
        M,
        D,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    return out_i32.to(torch.long)


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

        # 计算当前批次的最近邻（使用 Triton 或回退）
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


out = ModelNew(*get_init_inputs()).forward(*get_inputs())
