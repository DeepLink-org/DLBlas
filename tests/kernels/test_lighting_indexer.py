import torch
import pytest
from dlblas.kernels.lighting_indexer import mqa_attn_return_logits_interface
from dlblas.utils.utils import assert_tensors_similar, tensor_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple
from dlblas.utils.device_utils import infer_device, is_npu

device_ = torch.device(infer_device())

def display_error_message(msg):
    print(f"\033[31mWARNING: {msg}\033[0m")


def compute_correlation(a, b, label="tensor"):
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        display_error_message(f"{label} all zero")
        return 1
    correlation = 2 * (a * b).sum() / norm_sum
    return correlation


def validate_tensor_match(a, b, tolerance=1e-8, tensor_name="tensor", should_raise=True):
    a_finite = torch.isfinite(a)
    b_finite = torch.isfinite(b)
    if not torch.all(a_finite == b_finite):
        display_error_message(f"{tensor_name} Error: isfinite mask mismatch")
        if should_raise:
            assert False
    if not torch.isclose(
            a.masked_fill(a_finite, 0),
            b.masked_fill(b_finite, 0),
            rtol=0,
            atol=0,
            equal_nan=True,
    ).all():
        display_error_message(f"{tensor_name} Error: nonfinite value mismatch")
        if should_raise:
            assert False
    a = a.masked_fill(~a_finite, 0)
    b = b.masked_fill(~b_finite, 0)
    correlation = compute_correlation(a, b, tensor_name)
    difference = 1.0 - correlation
    if not (0 <= difference <= tolerance):
        display_error_message(f"{tensor_name} Error: {difference}")
        if should_raise:
            assert False
    return difference

def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
    k = kv
    q = q.float()
    k = k.float()

    seq_len_kv = kv.shape[0]
    mask_lo = torch.arange(0, seq_len_kv, device=device_)[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device=device_)[None, :] < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    score = torch.einsum('mhd,nd->hmn', q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float('-inf'))

    cost = mask.sum()
    return logits, cost, mask

def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple[int],
                                use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()

@tensor_cache
def cal_cu_seqlen_ke_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor,
                           cu_seqlens_ks: torch.LongTensor, cu_seqlens_ke: torch.LongTensor,
                           q_start_idxs: torch.LongTensor, seq_len: int,
                           kv_stride: int) -> torch.IntTensor:
    cu_seqlen_ke_for_each_q = torch.gather(
        input=torch.cat(
            [cu_seqlens_ke,
             torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]),
        dim=0,
        index=cal_seq_idx_for_q(
            cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long())
    casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,),
                                                 dtype=torch.int32,
                                                 device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = (torch.arange(
            q_start_idxs[i],
            q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i],
            dtype=torch.int32,
            device=cu_seqlens_qs.device) + 1) // kv_stride + cu_seqlens_ks[i]
    cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q, cu_seqlen_ke_for_each_q)
    return cu_seqlen_ke_for_each_q.int()


@tensor_cache
def cal_seq_idx_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor,
                      seq_len: int) -> torch.IntTensor:
    seq_idx_for_q = torch.full((seq_len,),
                               len(cu_seqlens_qs),
                               dtype=torch.int32,
                               device=cu_seqlens_qs.device)
    for i in range(len(cu_seqlens_qs)):
        seq_idx_for_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = i
    return seq_idx_for_q


@tensor_cache
def cal_cu_seqlen_ks_for_q(cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor,
                           cu_seqlens_ks: torch.LongTensor, seq_len: int) -> torch.IntTensor:
    cu_seqlen_ks_for_each_q = torch.gather(
        input=torch.cat([
            cu_seqlens_ks,
            torch.full((1,),
                       torch.iinfo(torch.int32).max,
                       dtype=torch.int32,
                       device=cu_seqlens_qs.device)
        ]),
        dim=0,
        index=cal_seq_idx_for_q(
            cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long())
    return cu_seqlen_ks_for_each_q.int()

def generate_random_cu_seqlens(per_cp_seqlen, cp_size=4, cp_rank=3, kv_stride=1, average_q_len=512):
    total_seqlen = per_cp_seqlen * cp_size

    cu_seqlens = torch.randint(0, average_q_len * 2, (total_seqlen // average_q_len * 2,), device=device_)
    last_seq_id = torch.where(cu_seqlens.cumsum(0) >= total_seqlen)[0][0]
    cu_seqlens = cu_seqlens[:last_seq_id]

    if cu_seqlens.sum() < total_seqlen:
        cu_seqlens = torch.cat([cu_seqlens, torch.tensor([total_seqlen - cu_seqlens.sum()], device=device_)])

    cu_seqlens_cumsum = torch.cumsum(cu_seqlens, dim=0)
    cu_seqlens_k_cumsum = torch.cumsum(cu_seqlens // kv_stride, dim=0)
    cu_seqlens_qs = torch.cat([torch.tensor([0], device=device_), cu_seqlens_cumsum[:-1]])
    cu_seqlens_ks = torch.cat([torch.tensor([0], device=device_), cu_seqlens_k_cumsum[:-1]])
    cu_seqlens_qe = cu_seqlens_cumsum.clone()
    cu_seqlens_ke = cu_seqlens_k_cumsum.clone()

    cu_seqlens_ks_for_each_q = cal_cu_seqlen_ks_for_q(
        cu_seqlens_qs=cu_seqlens_qs,
        cu_seqlens_qe=cu_seqlens_qe,
        cu_seqlens_ks=cu_seqlens_ks,
        seq_len=total_seqlen,
    )
    cu_seqlens_ke_for_each_q = cal_cu_seqlen_ke_for_q(
        cu_seqlens_qs=cu_seqlens_qs,
        cu_seqlens_qe=cu_seqlens_qe,
        cu_seqlens_ks=cu_seqlens_ks,
        cu_seqlens_ke=cu_seqlens_ke,
        q_start_idxs=torch.zeros_like(cu_seqlens_qs),
        seq_len=total_seqlen,
        kv_stride=kv_stride,
    )

    assert per_cp_seqlen % 2 == 0
    per_chunk_seqlen = per_cp_seqlen // 2
    slice_short = slice(cp_rank * per_chunk_seqlen, (cp_rank + 1) * per_chunk_seqlen)
    slice_long = slice(
        total_seqlen - (cp_rank + 1) * per_chunk_seqlen,
        total_seqlen - cp_rank * per_chunk_seqlen,
    )
    ks = torch.cat([
        cu_seqlens_ks_for_each_q[slice_short],
        cu_seqlens_ks_for_each_q[slice_long],
    ])
    ke = torch.cat([
        cu_seqlens_ke_for_each_q[slice_short],
        cu_seqlens_ke_for_each_q[slice_long],
    ])
    assert len(ks) == len(ke) == per_cp_seqlen
    return ks, ke



@pytest.mark.parametrize("S", [4096])
@pytest.mark.parametrize("SKV", [4096, 8192])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("HKV", [1])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("kv_stride", [1])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_fp8_lighting_indexer(S, SKV, H, HKV, D, kv_stride, dtype):
    if is_npu() and dtype == torch.float8_e4m3fn:
        pytest.skip("NPU not support fp8.")
    q = torch.randn(S, H, D, device=device_, dtype=torch.bfloat16)
    kv = torch.randn(SKV, D, device=device_, dtype=torch.bfloat16)
    weights = torch.randn(S, H, device=device_, dtype=torch.float32)
    p = (torch.randn(S, SKV, device=device_, dtype=torch.float32) * 4).softmax(dim=-1)

    ks, ke = generate_random_cu_seqlens(
        per_cp_seqlen=S, cp_size=4, cp_rank=3, kv_stride=kv_stride, average_q_len=2048)

    logits_ref, cost_ref, mask = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    if dtype == torch.float8_e4m3fn:
        q_fp8 = q.to(torch.float8_e4m3fn)
        kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
        logits_triton = mqa_attn_return_logits_interface(
            q=q_fp8.clone(), kv=kv_fp8.clone(), kv_scales=kv_scales.clone(), weights=weights.clone(), cu_seqlen_ks=ks.clone(), cu_seqlen_ke=ke.clone())
    else:
        logits_triton = mqa_attn_return_logits_interface(
            q=q, kv=kv, kv_scales=None, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke)

    logits_triton = logits_triton.masked_fill(~mask, float('-inf'))
    
    diff = validate_tensor_match(
        logits_triton, logits_ref, tolerance=1e-2, tensor_name="logits", should_raise=True)

    print(f"diff: {diff}")
