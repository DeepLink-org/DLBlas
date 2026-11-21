import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def grid_cluster_kernel(
    pos_ptr, size_ptr, start_ptr, stride_ptr, out_ptr,
    N, D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, BLOCK_D)

    mask_n = off_n < N
    mask_d = off_d < D

    # 二维坐标寻址，行优先布局
    pos_off = off_n[:, None] * D + off_d[None, :]
    pos = tl.load(pos_ptr + pos_off, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    # 一次性加载一维小向量，复用寄存器
    size = tl.load(size_ptr + off_d, mask=mask_d, other=1.0)
    start = tl.load(start_ptr + off_d, mask=mask_d, other=0.0)
    stride = tl.load(stride_ptr + off_d, mask=mask_d, other=1)

    # 网格索引计算 + 下界截断
    grid_idx = ((pos - start[None, :]) / size[None, :]).to(tl.int64)
    grid_idx = tl.maximum(grid_idx, 0)

    # 多维索引映射唯一一维ID
    cluster_id = tl.sum(grid_idx * stride[None, :], axis=1)
    tl.store(out_ptr + off_n, cluster_id, mask=mask_n)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos, size, start=None, end=None):
        # 维度校验
        if pos.dim() != 2:
            raise ValueError(f"pos should be 2-dimensional, got {pos.dim()}-dimensional")
        if size.dim() != 1:
            raise ValueError(f"size should be 1-dimensional, got {size.dim()}-dimensional")
        N, D = pos.shape
        if pos.size(1) != size.size(0):
            raise ValueError(f"Dimension mismatch: pos has {pos.size(1)} dimensions, "
                             f"but size has {size.size(0)} dimensions")
        device = pos.device

        # start 默认初始化
        if start is None:
            start = torch.zeros(D, device=device, dtype=pos.dtype)
        else:
            if start.dim() != 1 or start.size(0) != D:
                raise ValueError(f"start should have shape [{D}], got {start.shape}")
        # end 默认初始化
        if end is None:
            end = torch.max(pos, dim=0)[0] + size
        else:
            if end.dim() != 1 or end.size(0) != D:
                raise ValueError(f"end should have shape [{D}], got {end.shape}")

        # 按需转连续，避免不必要内存拷贝
        def ensure_contiguous(t):
            return t if t.is_contiguous() else t.contiguous()
        pos = ensure_contiguous(pos)
        size = ensure_contiguous(size)
        start = ensure_contiguous(start)
        end = ensure_contiguous(end)

        # 计算每个维度网格总数，规避浮点除法精度问题
        grid_cnt = torch.div((end - start), size, rounding_mode="trunc").long() + 1

        # stride 向量化预计算：cumprod + flip，避免 Python 循环
        if D == 1:
            stride = torch.ones_like(grid_cnt)
        else:
            stride = torch.cumprod(grid_cnt.flip(0), dim=0).flip(0) // grid_cnt
            stride[-1] = 1
        stride = ensure_contiguous(stride)

        cluster_id_buf = torch.empty(N, dtype=torch.long, device=device)
        grid = (triton.cdiv(N, 128),)

        grid_cluster_kernel[grid](
            pos, size, start, stride, cluster_id_buf,
            N, D,
            BLOCK_D=min(D, 4),
        )

        # 核心优化：跳过 torch.unique，直接返回原始网格编码ID
        return cluster_id_buf


def get_inputs():
    # pos = torch.tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
    # size = torch.tensor([5, 5])
    # end = torch.tensor([19, 19])
    pos = torch.tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]], dtype=torch.float32)
    size = torch.tensor([5, 5], dtype=torch.float32)
    end = torch.tensor([19, 19], dtype=torch.float32)
    N, D = 100000, 2
    torch.manual_seed(42)
    pos = torch.rand(N, D, dtype=torch.float32) * 100.0
    size = torch.full((D,), 5.0, dtype=torch.float32)
    end = torch.full((D,), 100.0, dtype=torch.float32)
  
    return [pos, size, end]


def get_init_inputs():
    return []


if __name__ == "__main__":
    dev = torch.device("npu:0")
    torch_npu.npu.set_device(0)
    model = ModelNew().to(dev)
    model.eval()
    pos, size, end = [x.to(dev) for x in get_inputs()]

    warmup_iters = 10
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(pos, size, end=end)

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./grid_cluster_prof")
    ) as prof:
        iter_num = 20
        for _ in range(iter_num):
            with torch.no_grad():
                out = model(pos, size, end=end)

    print("=" * 70)
    print(prof.key_averages().table(sort_by="npu_time_total", row_limit=20))
    print("输出结果：", out.cpu())