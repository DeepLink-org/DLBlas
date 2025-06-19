import torch
import triton
import triton.language as tl

from dlblas.utils import SymVar, Tensor, register_dlblas_op


@triton.jit
def _topk_gating_fwd_part3_rearrange_opt(
    gates,
    locations,
    masks,
    invers_k_ptr,
    token_rearranged_ec_idx_ptr,
    gate_ks_ptr,
    token_sel_exp_int_mask_ptr,
    stride_ks_k,
    stride_se_s,
    stride_kse_k,
    stride_kse_s,
    CAPACITY: tl.constexpr,
    min_value: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    s_pid = tl.program_id(axis=0)
    offs_e = tl.arange(0, EXPERTS)[None, :]
    locations_ptrs = locations + offs_k[:, None] * stride_kse_k + s_pid * stride_kse_s + offs_e
    locations_data = tl.load(locations_ptrs, mask=offs_k[:, None] < K)
    masks_ptrs = masks + offs_k[:, None] * stride_kse_k + s_pid * stride_kse_s + offs_e
    masks_data = tl.load(masks_ptrs, mask=offs_k[:, None] < K)
    masks_data *= tl.where(locations_data < CAPACITY, 1, 0)
    # for bwd
    tl.store(masks_ptrs, masks_data, mask=offs_k[:, None] < K)
    locations_data *= masks_data
    locations_ks_data = tl.sum(locations_data, axis=1)
    gates_ptrs = gates + s_pid * stride_se_s + offs_e
    gates_data = tl.load(gates_ptrs)
    # gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    multi = tl.broadcast_to(gates_data, (BLOCK_K, EXPERTS)) * masks_data
    # gates_s = tl.sum(multi, axis=1)
    gates_ks, indices_ks = tl.max(multi, axis=1, return_indices=True)
    denom_s = tl.sum(gates_ks, axis=0)
    # torch.clamp
    denom_s = tl.where(denom_s < min_value, min_value, denom_s)
    # ks
    gates_ks /= denom_s
    gate_ks_ptrs = gate_ks_ptr + offs_k * stride_ks_k + s_pid
    tl.store(gate_ks_ptrs, gates_ks, mask=offs_k < K)
    token_rearranged_ec_idx_data = indices_ks.to(tl.int32) * CAPACITY + locations_ks_data.to(tl.int32)
    token_rearranged_ec_idx_ptrs = token_rearranged_ec_idx_ptr + offs_k * stride_ks_k + s_pid
    tl.store(token_rearranged_ec_idx_ptrs, token_rearranged_ec_idx_data, mask=offs_k < K)
    # se
    invers_k_data = tl.load(invers_k_ptr + offs_k, mask=offs_k < K)
    token_sel_exp_int_mask_data = masks_data * tl.broadcast_to(tl.expand_dims(invers_k_data, axis=1),
                                                               (BLOCK_K, EXPERTS))
    token_sel_exp_int_mask_data = tl.sum(token_sel_exp_int_mask_data, axis=0)
    token_sel_exp_int_mask_ptrs = token_sel_exp_int_mask_ptr + s_pid * stride_se_s + tl.arange(0, EXPERTS)
    tl.store(token_sel_exp_int_mask_ptrs, token_sel_exp_int_mask_data)


@triton.jit
def _topk_gating_fwd_part3(
    gates,
    gates_all,
    locations,
    locations_s,
    masks,
    combine_weights,
    dispatch_mask,
    stride_ks_k,
    stride_se_s,
    stride_kse_k,
    stride_kse_s,
    stride_sec_s,
    stride_sec_e,
    CAPACITY: tl.constexpr,
    BLOCK_C: tl.constexpr,
    min_value: tl.constexpr,
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)[:, None]
    s_pid = tl.program_id(axis=0)
    offs_e = tl.arange(0, EXPERTS)[None, :]
    locations_ptrs = locations + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    locations_data = tl.load(locations_ptrs, mask=offs_k < K)
    masks_ptrs = masks + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    masks_data = tl.load(masks_ptrs, mask=offs_k < K)
    masks_data *= tl.where(locations_data < CAPACITY, 1, 0)
    # for bwd
    tl.store(masks_ptrs, masks_data, mask=offs_k < K)
    locations_data *= masks_data
    locations_s_data = tl.reshape(tl.sum(locations_data, axis=1), (BLOCK_K, 1))
    gates_ptrs = gates + s_pid * stride_se_s + offs_e
    gates_data = tl.load(gates_ptrs)
    # gate_s = torch.einsum("se,kse->ks", gates, mask_float)
    multi = tl.broadcast_to(gates_data, (BLOCK_K, EXPERTS)) * masks_data
    gates_s = tl.sum(multi, axis=1)
    denom_s = tl.sum(gates_s, axis=0)
    # torch.clamp
    denom_s = tl.where(denom_s < min_value, min_value, denom_s)
    gates_s /= denom_s
    gates_s = tl.reshape(gates_s, (BLOCK_K, 1))
    gates_all_data = tl.broadcast_to(gates_s, (BLOCK_K, EXPERTS)) * masks_data.to(gates_s.dtype)
    # test
    gates_all_ptrs = gates_all + offs_k * stride_kse_k + s_pid * stride_kse_s + offs_e
    tl.store(gates_all_ptrs, gates_all_data, mask=offs_k < K)
    locas_s_ptrs = locations_s + offs_k * stride_ks_k + s_pid
    tl.store(locas_s_ptrs, locations_s_data, mask=offs_k < K)


def call(gates, masks, locations, k, capacity, enable_token_rearrange_opt):
    if enable_token_rearrange_opt:
        return rearrange_opt(gates, masks, locations, k, capacity)
    else:
        return topk_gating_fwd_part3(gates, masks, locations, k, capacity)


def rearrange_opt(gates, masks, locations, k, capacity):
    s, e = gates.shape
    invers_k = torch.arange(k, 0, -1, device=masks.device)
    token_rearranged_ec_idx = torch.empty((k, s), dtype=torch.int32, device=gates.device)
    gate_ks = torch.empty((k, s), dtype=gates.dtype, device=gates.device)
    token_sel_exp_int_mask = torch.empty((s, e), dtype=torch.int64, device=gates.device)
    stride_se_s, _ = gates.stride()
    stride_ks_k, _ = gate_ks.stride()
    stride_kse_k, stride_kse_s, _ = masks.stride()
    with torch.cuda.device(gates.device):
        _topk_gating_fwd_part3_rearrange_opt[(s, )](
            gates,
            locations,
            masks,
            invers_k,
            token_rearranged_ec_idx,
            gate_ks,
            token_sel_exp_int_mask,
            stride_ks_k,
            stride_se_s,
            stride_kse_k,
            stride_kse_s,
            CAPACITY=capacity,
            min_value=torch.finfo(gates.dtype).eps,
            K=k,
            EXPERTS=e,
            BLOCK_K=triton.next_power_of_2(k),
        )
    expert_sel_top_c_token_idx = torch.topk(token_sel_exp_int_mask, k=capacity, dim=0, sorted=True)[1]
    expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(e * capacity)
    token_rearranged_ec_idx = token_rearranged_ec_idx.reshape(-1)
    token_exp_weights = gate_ks.reshape(-1)
    return (token_rearranged_ec_idx, token_exp_weights, expert_select_token_idx)


def topk_gating_fwd_part3(gates, masks, locations, k, capacity):
    s, e = gates.shape
    stride_se_s, _ = gates.stride()
    gates_all = torch.empty((k, s, e), dtype=gates.dtype, device=gates.device)
    locations_s = torch.empty((k, s), dtype=torch.int64, device=gates.device)
    combine_weights = torch.empty((s, e, capacity), device=gates.device)
    dispatch_mask = torch.empty((s, e, capacity), device=gates.device, dtype=torch.bool)
    stride_sec_s, stride_sec_e, _ = combine_weights.stride()
    stride_kse_k, stride_kse_s, _ = masks.stride()
    min_value = torch.finfo(gates.dtype).eps
    with torch.cuda.device(gates.device):
        _topk_gating_fwd_part3[(s, )](
            gates,
            gates_all,
            locations,
            locations_s,
            masks,
            combine_weights,
            dispatch_mask,
            stride_ks_k=s,
            stride_se_s=stride_se_s,
            stride_kse_k=stride_kse_k,
            stride_kse_s=stride_kse_s,
            stride_sec_s=stride_sec_s,
            stride_sec_e=stride_sec_e,
            CAPACITY=capacity,
            BLOCK_C=triton.next_power_of_2(capacity),
            min_value=min_value,
            K=k,
            EXPERTS=e,
            BLOCK_K=triton.next_power_of_2(k),
        )
    locations_sc = torch.nn.functional.one_hot(locations_s, num_classes=capacity)
    combine_sec = torch.einsum('kse,ksc->ksec', gates_all, locations_sc)
    combine_weights = torch.sum(combine_sec, dim=0)
    dispatch_mask = combine_weights.bool()
    return combine_weights, dispatch_mask


def bench_fn(gates, masks, locations, k, capacity, enable_token_rearrange_opt):
    ms = triton.testing.do_bench(lambda: call(gates, masks, locations, k, capacity, enable_token_rearrange_opt),
                                 warmup=10,
                                 rep=10)
    return ms


# register
name = '_topk_gating_fwd_part3'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k, capacity = SymVar('k'), SymVar('capacity')
        # we dont' actually allocate tensor
        gates = Tensor((seqLen, experts), dtype=dtype, device=device)
        masks = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        locations = Tensor((k, seqLen, experts), dtype=torch.int64, device=device)
        register_dlblas_op(name, None, (gates, masks, locations, torch.SymInt, torch.SymInt, torch.SymBool), call,
                           bench_fn, call)
