import torch

import triton
import triton.language as tl


@triton.jit
def _pairwise_distance_kernel(
    X,
    Y,
    D,
    N,
    M,
    F,
    stride_xn,
    stride_xf,
    stride_ym,
    stride_yf,
    stride_dn,
    stride_dm,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Compute pairwise squared Euclidean distance between rows of X [N, F] and Y [M, F]:
      D[n, m] = ||X[n] - Y[m]||^2
    Implemented as: ||x||^2 + ||y||^2 - 2 * x·y
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Accumulator for dot products and squared norms
    acc_xy = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    x_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    y_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over feature dimension
    for fk in range(0, F, BLOCK_F):
        offs_f = fk + tl.arange(0, BLOCK_F)

        # X tile: [BLOCK_N, BLOCK_F]
        x_ptrs = X + offs_n[:, None] * stride_xn + offs_f[None, :] * stride_xf
        x_mask = (offs_n[:, None] < N) & (offs_f[None, :] < F)
        x_sub = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Y tile transposed: we construct [BLOCK_F, BLOCK_M] for dot
        y_ptrs_T = Y + offs_m[None, :] * stride_ym + offs_f[:, None] * stride_yf
        y_mask_T = (offs_m[None, :] < M) & (offs_f[:, None] < F)
        y_sub_T = tl.load(y_ptrs_T, mask=y_mask_T, other=0.0)

        # Accumulate dot products and squared norms
        acc_xy += tl.dot(x_sub, y_sub_T)  # [BLOCK_N, BLOCK_M]
        x_sq += tl.sum(x_sub * x_sub, axis=1)  # [BLOCK_N]
        y_sq += tl.sum(y_sub_T * y_sub_T, axis=0)  # [BLOCK_M]

    # Distance = ||x||^2 + ||y||^2 - 2*x·y
    dist = x_sq[:, None] + y_sq[None, :] - 2.0 * acc_xy
    # Numerical stability: clamp to >= 0
    dist = tl.maximum(dist, 0.0)

    # Store with bounds mask
    d_ptrs = D + offs_n[:, None] * stride_dn + offs_m[None, :] * stride_dm
    mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(d_ptrs, dist, mask=mask)


@triton.jit
def _pairwise_dot_kernel(
    X,
    Y,
    S,
    N,
    M,
    F,
    stride_xn,
    stride_xf,
    stride_ym,
    stride_yf,
    stride_sn,
    stride_sm,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Compute pairwise dot products between rows of X [N, F] and Y [M, F]:
      S[n, m] = X[n] · Y[m]
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc_xy = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for fk in range(0, F, BLOCK_F):
        offs_f = fk + tl.arange(0, BLOCK_F)

        # X tile: [BLOCK_N, BLOCK_F]
        x_ptrs = X + offs_n[:, None] * stride_xn + offs_f[None, :] * stride_xf
        x_mask = (offs_n[:, None] < N) & (offs_f[None, :] < F)
        x_sub = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Y tile transposed: [BLOCK_F, BLOCK_M]
        y_ptrs_T = Y + offs_m[None, :] * stride_ym + offs_f[:, None] * stride_yf
        y_mask_T = (offs_m[None, :] < M) & (offs_f[:, None] < F)
        y_sub_T = tl.load(y_ptrs_T, mask=y_mask_T, other=0.0)

        acc_xy += tl.dot(x_sub, y_sub_T)

    # Store with bounds mask
    s_ptrs = S + offs_n[:, None] * stride_sn + offs_m[None, :] * stride_sm
    mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(s_ptrs, acc_xy, mask=mask)


def _launch_pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernel to compute pairwise squared Euclidean distances
    between rows of x [N, F] and y [M, F]. Returns [N, M].
    """
    assert x.is_cuda and y.is_cuda, "Triton path requires CUDA tensors"
    N, F = x.shape
    M, _ = y.shape
    # Ensure contiguous for simple stride semantics
    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty((N, M), device=x.device, dtype=torch.float32)

    # Tile sizes tuned for small-to-medium problems
    BLOCK_N = 32  # 64
    BLOCK_M = 32  # 64
    BLOCK_F = 16  # 32

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _pairwise_distance_kernel[grid](
        x_c,
        y_c,
        out,
        N,
        M,
        F,
        x_c.stride(0),
        x_c.stride(1),
        y_c.stride(0),
        y_c.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_F=BLOCK_F,
        num_warps=4,
        num_stages=2,
    )
    return out


def _launch_pairwise_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Launch Triton kernel to compute pairwise dot products x @ y^T.
    x: [N, F], y: [M, F] -> out: [N, M]
    """
    # assert x.is_cuda and y.is_cuda, "Triton path requires CUDA tensors"
    N, F = x.shape
    M, _ = y.shape
    x_c = x.contiguous()
    y_c = y.contiguous()
    out = torch.empty((N, M), device=x.device, dtype=torch.float32)

    BLOCK_N = 64
    BLOCK_M = 64
    BLOCK_F = 32

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _pairwise_dot_kernel[grid](
        x_c,
        y_c,
        out,
        N,
        M,
        F,
        x_c.stride(0),
        x_c.stride(1),
        y_c.stride(0),
        y_c.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_F=BLOCK_F,
        num_warps=4,
        num_stages=2,
    )
    return out


def _knn_single_batch(x, y, k, cosine=False):
    """
    单批次的knn计算（内部函数）
    """
    N, F = x.shape
    M, _ = y.shape

    # Fallback to PyTorch if tensors are not on CUDA
    use_triton = True  # x.is_cuda and y.is_cuda

    if cosine:
        # 使用余弦相似度
        # 归一化向量
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)

        # 计算余弦相似度（越大越相似）
        if use_triton:
            similarity = _launch_pairwise_dot(x_norm, y_norm)
        else:
            similarity = torch.matmul(x_norm, y_norm.t())

        # 获取top-k相似度的索引（largest=True）
        topk_values, topk_indices = torch.topk(
            similarity, k=min(k, M), dim=1, largest=True
        )

    else:
        # 使用欧氏距离
        if use_triton:
            # 直接用自定义核计算距离矩阵并钳制到非负
            distance = _launch_pairwise_distance(x, y)
        else:
            # 计算距离矩阵: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            x_squared = (x**2).sum(dim=1, keepdim=True)  # [N, 1]
            y_squared = (y**2).sum(dim=1, keepdim=True)  # [M, 1]
            xy = torch.matmul(x, y.t())  # [N, M]
            distance = x_squared + y_squared.t() - 2 * xy  # [N, M]
            distance = torch.clamp(distance, min=0)

        # 获取top-k最小距离的索引
        topk_values, topk_indices = torch.topk(
            distance, k=min(k, M), dim=1, largest=False
        )

    # 构建row和col索引（保持与原实现一致）
    row = torch.arange(N, device=x.device).repeat_interleave(k)  # [N*k]
    col = topk_indices.flatten()  # [N*min(k, M)]

    return row, col


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

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
out = ModelNew(*get_init_inputs()).forward(*get_inputs())
print(out)
