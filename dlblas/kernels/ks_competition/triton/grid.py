import torch
import triton
import triton.language as tl


@triton.jit
def compute_cluster_ids_kernel(
    pos_ptr,  # * pointer to input positions [N, D]
    start_ptr,  # * pointer to start tensor [D]
    size_ptr,  # * pointer to size tensor [D]
    grid_counts_ptr,  # * pointer to grid_counts tensor [D]
    out_ptr,  # * pointer to output cluster_ids [N]
    N,  # number of points
    stride_pos_row,  # stride between rows in pos (in elements)
    stride_pos_col,  # stride between columns in pos (in elements)
    D: tl.constexpr,  # dimensionality
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    # base pointer offsets for each row
    base = offs * stride_pos_row
    row_ptr = pos_ptr + base

    # initialize accumulator for cluster id
    acc = tl.zeros([BLOCK_N], dtype=tl.int64)

    # Prefetch first dimension's positions to hide some latency
    pos_next = tl.load(row_ptr + 0 * stride_pos_col, mask=mask, other=0)

    # Iterate over dimensions and compute flattened cluster id
    for d in tl.static_range(0, D):
        pos_vals = pos_next
        # Prefetch next dimension if exists
        if d < D - 1:
            pos_next = tl.load(row_ptr + (d + 1) * stride_pos_col, mask=mask, other=0)

        # Load scalars for this dimension and compute reciprocal for faster math
        s_f = tl.load(start_ptr + d).to(tl.float32)
        sz_f = tl.load(size_ptr + d).to(tl.float32)
        inv_sz = 1.0 / sz_f

        # Convert to float for division
        pos_f = pos_vals.to(tl.float32)

        # Compute grid index for this dimension: ((pos - start) / size).long(), clamped at min=0
        div = (pos_f - s_f) * inv_sz
        g = div.to(tl.int64)
        g = tl.maximum(g, 0)

        # Accumulate into unique id: on first dim, just set; otherwise Horner-like accumulation
        if d == 0:
            acc = g
        else:
            cnt = tl.load(grid_counts_ptr + d).to(tl.int64)
            acc = acc * cnt + g

    # Store result
    tl.store(out_ptr + offs, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

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

        # 若在CUDA上，使用triton kernel加速cluster_ids计算
        if pos.is_cuda:
            # 计算每个维度上的网格数量，与原始实现保持一致
            grid_counts = ((end - start) / size).long() + 1

            # 输出cluster_ids
            cluster_ids = torch.empty(N, dtype=torch.long, device=device)

            # Launch Triton kernel
            BLOCK_N = 1024
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
            compute_cluster_ids_kernel[grid](
                pos,
                start,
                size,
                grid_counts,
                cluster_ids,
                N,
                pos.stride(0),
                pos.stride(1),
                D=D,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )

            # 重新映射聚类ID为连续的整数，与原始实现一致
            unique_ids, inverse_indices = torch.unique(cluster_ids, return_inverse=True)
            return inverse_indices

        # CPU或其他设备上保持原始实现
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
out = ModelNew(*get_init_inputs()).forward(*get_inputs())
print(out)
