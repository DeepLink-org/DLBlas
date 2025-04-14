import torch
import triton
import triton.language as tl
from packaging import version

if version.parse(triton.__version__) <= version.parse('2.2.0'):

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        from triton.runtime.jit import get_cuda_stream

        device = tensor.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)
else:

    KERNEL_META = dict()

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        return KERNEL_META


@triton.jit
def _start_end_kernel(TopkIdx, SortedIdx, ExpStart, ExpEnd, len_sorted_idx: int, num_experts: tl.constexpr,
                      BLOCK: tl.constexpr):
    """start end kernel."""
    exp_id = tl.program_id(0)
    exp_start = -1
    cnt = 0

    s_off = tl.arange(0, BLOCK)

    # find start
    for sidx_start in range(0, len_sorted_idx, BLOCK):
        sidx_off = sidx_start + s_off
        sidx_mask = sidx_off < len_sorted_idx
        sidx = tl.load(SortedIdx + sidx_off, mask=sidx_mask, other=0)
        tidx = tl.load(TopkIdx + sidx, mask=sidx_mask, other=num_experts)
        tidx_mask = tidx == exp_id
        cnt += tl.sum(tidx_mask.to(tl.int32))
        if cnt > 0 and exp_start < 0:
            exp_start = sidx_start + tl.argmax(tidx_mask, axis=0)

    if exp_start < 0:
        exp_start *= 0
    exp_end = exp_start + cnt
    tl.store(ExpStart + exp_id, exp_start)
    tl.store(ExpEnd + exp_id, exp_end)


def dlblas_get_start_end(topk_idx: torch.Tensor, sorted_idx: torch.Tensor, num_experts: int):
    """get start and end.
    same process as:
    >>> exp_tok_cnt = F.one_hot(flatten_topk_ids, num_classes=E).sum(0)
    >>> exp_end = exp_tok_cnt.cumsum(0)
    >>> exp_start = exp_end - exp_tok_cnt
    """
    start_end = sorted_idx.new_empty(2, num_experts)
    exp_start = start_end[0, :]
    exp_end = start_end[1, :]

    BLOCK = 128
    kernel_meta = get_kernel_meta(topk_idx)
    _start_end_kernel[(num_experts, )](
        topk_idx,
        sorted_idx,
        exp_start,
        exp_end,
        len_sorted_idx=sorted_idx.numel(),
        num_experts=num_experts,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=1,
        **kernel_meta,
    )

    return exp_start, exp_end


def _dlblas_get_sorted_idx(topk_ids: torch.Tensor, num_experts: int):
    """get sorted idx."""
    flatten_topk_ids = topk_ids.flatten()
    sorted_idx = flatten_topk_ids.argsort()
    exp_start, exp_end = dlblas_get_start_end(flatten_topk_ids, sorted_idx, num_experts)
    return sorted_idx, exp_start, exp_end
