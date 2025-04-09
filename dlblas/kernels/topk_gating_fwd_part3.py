import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace

    
@triton.jit
def _topk_gating_fwd_part3(
    mask_with_capacity,
    topk_indices_ptr,
    topk_probs_ptr, 
    final_indices_ptr,
    final_probs_ptr,
    fill_value,
    stride_ks_k,
    stride_se_s,
    CAPACITY: tl.constexpr,
    BLOCK_C: tl.constexpr,
    min_value: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)[None,:]
    s_pid = tl.program_id(axis=0)
    offs_e = tl.arange(0, EXPERTS)[None,:]
    

    mask_with_capacity_ptrs = mask_with_capacity + s_pid * stride_se_s + offs_e
    final_mask = tl.load(mask_with_capacity_ptrs, mask=offs_e < EXPERTS) # (s, e)对应torch中的final_mask(多的选择出来少的不补)
    drop_mask = tl.where(final_mask == 0, 1, 0)

    # tl.device_print("drop_mask.shape[0]", drop_mask.shape[0])
    # tl.device_print("drop_mask.shape[1]", drop_mask.shape[1])

    topk_indices_ptrs = topk_indices_ptr + s_pid * K + offs_k
    topk_indices = tl.load(topk_indices_ptrs, mask=offs_k < K)

    topk_probs_ptrs = topk_probs_ptr + s_pid * K + offs_k
    topk_probs = tl.load(topk_probs_ptrs, mask=offs_k < K)

    # Gather drop_mask based on topk_indices (Torch equivalent: exceed_mask = torch.gather(drop_mask, 1, top_indices))
    gathered_drop_mask = tl.zeros((1, K), dtype=tl.int32)
    for idx in tlc.static_range(K):
        idx_offset = tl.sum((tl.arange(0, K)[None, :] == idx) * topk_indices, axis = 1)
        drop_mask_data = tl.sum((tl.arange(0, EXPERTS)[None, :] == idx_offset) * drop_mask, axis = 1)
        gathered_drop_mask += (tl.arange(0, K)[None, :] == idx) * drop_mask_data

    # Compute exceed_mask (Torch equivalent: exceed_mask = torch.gather(drop_mask, 1, top_indices))
    exceed_mask = tl.where(gathered_drop_mask > 0, 1, 0)
    final_probs = topk_probs * (1 - exceed_mask)
    final_probs_ptrs = final_probs_ptr + s_pid * K + offs_k
    tl.store(final_probs_ptrs, final_probs, mask=offs_k < K)

    final_indices = tl.where(exceed_mask > 0, fill_value, topk_indices)
    final_indices_ptrs = final_indices_ptr + s_pid * K + offs_k
    tl.store(final_indices_ptrs, final_indices, mask=offs_k < K)
    
def call(gates, mask_with_capacity, topk_indices, topk_values, k, capacity):
    fill_value = torch.finfo(gates.dtype).max
    s, e = gates.shape
    stride_se_s, _ = gates.stride()
    final_indices = torch.empty((s, k), dtype=torch.int64, device=gates.device)
    final_probs = torch.empty((s, k), dtype=topk_values.dtype, device=gates.device)
    min_value = torch.finfo(gates.dtype).eps
    with torch.cuda.device(gates.device):
        _topk_gating_fwd_part3[(s,)](
            mask_with_capacity,
            topk_indices, 
            topk_values, 
            final_indices,
            final_probs,
            fill_value,
            stride_ks_k=s,
            stride_se_s=stride_se_s,
            CAPACITY = capacity,
            BLOCK_C = triton.next_power_of_2(capacity),
            min_value = min_value,
            K = k,
            EXPERTS = e,
            BLOCK_K = triton.next_power_of_2(k),
        )
    return final_indices, final_probs


def bench_fn(gates, mask_with_capacity, topk_indices, topk_values, k, capacity):
    fn = lambda: call(gates, mask_with_capacity, topk_indices, topk_values, k, capacity)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_fwd_part3'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k, capacity = SymVar('k'), SymVar('capacity')
        gates = Tensor((seqLen, experts), dtype=dtype, device=device)
        mask_with_capacity = Tensor((seqLen, experts), dtype=dtype, device=device) # locations = masks_with_capacity
        topk_indices = Tensor((seqLen, k), dtype=torch.int64, device=device)
        topk_values = Tensor((seqLen, k), dtype=dtype, device=device)
        register_dlblas_op(name, None, (gates, mask_with_capacity, topk_indices, topk_values, torch.SymInt, torch.SymInt), call, bench_fn, _topk_gating_fwd_part3)
