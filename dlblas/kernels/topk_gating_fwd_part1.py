import torch
import triton
import triton.language as tl
import triton.language.core as tlc

from dlblas.utils import ChoiceSpace, SymVar, Tensor, register_dlblas_op

if triton.__version__ >= '3.0.0':
    from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
else:
    from triton.language.math import fast_expf as tl_exp


@triton.autotune(
    configs = [
        triton.Config({'BLOCK_S': BS}, num_stages=7, num_warps=4, num_ctas=2) \
        for BS in [1] \
    ],
    key=['seq_len'],
)
@triton.jit
def _topk_gating_kernel_part1(
    logits_ptr,
    scores_ptr,
    masks_ptr,  # output
    masks_gates_ptr,
    top_indices_ptr,
    top_values_ptr,
    fill_value,
    stride_se_s,
    stride_sk_s,
    seq_len,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EXPERTS: tl.constexpr,
):
    # the softmax computation for each row is independent
    # each block process each row
    pid = tl.program_id(axis=0)

    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    offs_e = tl.arange(0, EXPERTS)[None, :]
    offs_k = tl.arange(0, K)[None, :]

    # load data
    logits_ptrs = logits_ptr + offs_s * stride_se_s + offs_e
    # logits_ptrs = tl.make_block_ptr(logits_ptr,shape=(BLOCK_S, EXPERTS),
    #                                            strides=(stride_se_s, 1),
    #                                            offsets=(pid * BLOCK_S, 0),
    #                                            block_shape=(BLOCK_S, EXPERTS),
    #                                            order=(1, 0))
    logits_data = tl.load(logits_ptrs, mask=offs_s < seq_len)

    # Apply row-wise softmax to values_data
    max_values = tl.max(logits_data, axis=1)[:, None]
    sub_values = logits_data - max_values  # Subtract max for numerical stability
    exp_values = tl.exp(sub_values)
    sum_exp = tl.sum(exp_values, axis=1)[:, None]
    scores_data = exp_values / sum_exp  # Normalize to compute softmax

    scores_ptrs = scores_ptr + offs_s * stride_se_s + offs_e
    tl.store(scores_ptrs, scores_data, mask=offs_s < seq_len)

    # create mask
    masks_data = tl.zeros((BLOCK_S, EXPERTS), tl.int64)
    masks_gates_data = tl.zeros((BLOCK_S, EXPERTS), tl.int64)

    # prepare indices storage
    indices_data = tl.zeros((BLOCK_S, K), dtype=tl.int64)
    values_data = tl.zeros((BLOCK_S, K), dtype=logits_data.dtype)

    # select topk based on logits
    for idx in tlc.static_range(K):
        # max_idx.shape = (BLOCK_S)
        max_idx = tl.argmax(logits_data, axis=1, tie_break_left=False)

        all_ids = tl.broadcast_to(tl.arange(0, EXPERTS)[None, :], (BLOCK_S, EXPERTS))
        max_idx_expand = tl.broadcast_to(tl.expand_dims(max_idx, axis=1), (BLOCK_S, EXPERTS))
        mask_data = tl.where(all_ids == max_idx_expand, 1, 0)
        max_vals = mask_data * logits_data
        # max_idx.shape = (BLOCK_S)
        max_vals = tl.sum(max_vals, axis=1)  # 每行的最大值 (BLOCK_S)
        masks_data += mask_data

        # Create mask for current column
        mask = tl.broadcast_to(tl.arange(0, K)[None, :] == idx, (BLOCK_S, K))  # (BLOCK_S, K)
        indices_data += mask * tl.broadcast_to(tl.expand_dims(max_idx, axis=1), (BLOCK_S, K))
        values_data += mask * tl.broadcast_to(tl.expand_dims(max_vals, axis=1), (BLOCK_S, K))

        logits_data = tl.where(mask_data > 0, fill_value, logits_data)

    # Apply row-wise softmax to values_data
    max_values = tl.max(values_data, axis=1)[:, None]
    values_data = values_data - max_values  # Subtract max for numerical stability
    exp_values = tl.exp(values_data)
    sum_exp = tl.sum(exp_values, axis=1)[:, None]
    values_data = exp_values / sum_exp  # Normalize to compute softmax

    # store mask_gates
    masks_gates = tl.zeros((BLOCK_S, EXPERTS), dtype=logits_data.dtype)
    for idx in tlc.static_range(K):
        all_ids = tl.broadcast_to(tl.arange(0, EXPERTS)[None, :], (BLOCK_S, EXPERTS))
        indices_with_idx = tl.broadcast_to(
            tl.expand_dims(tl.sum((tl.broadcast_to(tl.arange(0, K)[None, :], (BLOCK_S, K)) == idx) * indices_data,
                                  axis=1),
                           axis=1), (BLOCK_S, EXPERTS))
        values_with_idx = tl.broadcast_to(
            tl.expand_dims(tl.sum((tl.broadcast_to(tl.arange(0, K)[None, :], (BLOCK_S, K)) == idx) * values_data,
                                  axis=1),
                           axis=1), (BLOCK_S, EXPERTS))
        mask_data = tl.where(all_ids == indices_with_idx, 1, 0)
        mask_gates = mask_data * values_with_idx
        masks_gates += mask_gates
        # mask_data = tl.where(all_ids == indices_data[idx])
    masks_gates_ptrs = masks_gates_ptr + offs_s * stride_se_s + offs_e
    tl.store(masks_gates_ptrs, masks_gates, mask=offs_s < seq_len)

    # store mask
    masks_ptrs = masks_ptr + offs_s * stride_se_s + offs_e
    tl.store(masks_ptrs, masks_data, mask=offs_s < seq_len)

    # store topk indices
    top_indices_ptrs = top_indices_ptr + offs_s * stride_sk_s + offs_k
    tl.store(top_indices_ptrs, indices_data, mask=offs_s < seq_len)

    # store topk values
    top_values_ptrs = top_values_ptr + offs_s * stride_sk_s + offs_k
    tl.store(top_values_ptrs, values_data, mask=offs_s < seq_len)


def call(logits: torch.Tensor, k: int):
    # print("Logits shape:", logits.shape)
    # print("k value:", k)
    s, e = logits.shape
    stride_se_s, _ = logits.stride()
    stride_sk_s = k
    topk_mask = torch.empty((s, e), dtype=torch.int64, device=logits.device)
    topk_masked_gates = torch.empty((s, e), dtype=logits.dtype, device=logits.device)
    scores = torch.empty((s, e), dtype=logits.dtype, device=logits.device)
    top_values = torch.empty((s, k), dtype=logits.dtype, device=logits.device)
    top_indices = torch.empty((s, k), dtype=torch.int64, device=logits.device)

    fill_value = torch.finfo(logits.dtype).min
    grid = lambda META: (triton.cdiv(s, META['BLOCK_S']), )
    #     grid = lambda META: (
    #     META['num_ctas'],  # 设置网格的 CTA 数量
    #     triton.cdiv(s, META["BLOCK_S"])  # 网格的分块策略
    # )
    with torch.cuda.device(logits.device):
        _topk_gating_kernel_part1[grid](
            logits,
            scores,
            topk_mask,
            topk_masked_gates,
            top_indices,
            top_values,
            fill_value,
            stride_se_s=stride_se_s,
            stride_sk_s=stride_sk_s,
            seq_len=s,
            K=k,
            EXPERTS=e,
        )

    return scores, topk_mask, topk_masked_gates, top_indices, top_values


def bench_fn(logits: torch.Tensor, k: int):
    fn = lambda: call(logits, k)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_fwd_part1'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)
        register_dlblas_op(name, None, (logits, torch.SymInt), call, bench_fn, _topk_gating_kernel_part1)


def main():
    # triton.runtime.set_debug(True)
    # 参数设置
    SeqLen = 4096
    num_experts = 64
    k = 8

    # 初始化输入数据
    logits = torch.randn((SeqLen, num_experts), dtype=torch.float32, device='cuda')
    print('Input logits shape:', logits.shape)

    # 调用 kernel
    scores, masks, topk_masked_gates, top_indices, top_values = call(logits, k)

    # 打印输出结果
    print('Scores shape:', scores.shape)
    print('Masks shape:', masks.shape)
    print('Top indices shape:', top_indices.shape)
    print('Top values shape:', top_values.shape)


if __name__ == '__main__':
    main()
