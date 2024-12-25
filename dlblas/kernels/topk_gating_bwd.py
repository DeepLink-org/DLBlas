import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor
from dlblas.utils.libentry import libentry

@libentry()
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': BS}, num_stages=s, num_warps=w, num_ctas=c)
        for BS in [16, 32, 64]  # Possible block sizes
        for s in [1]
        for w in [1]
        for c in [1, 2, 4]
    ],
    key=['s', 'e', 'k'],
)

@triton.jit
def _topk_gating_bwd_kernel(
    tokens_per_expert_ptr,  # sum_1
    logits_softmax_ptr,     # _softmax_1
    grad_l_aux_ptr,         # tangents_1
    diag_mask_ptr,
    grad_logits_ptr,                # Result of softmax gradient
    stride_se_s,
    s: tl.constexpr,
    e: tl.constexpr,
    k: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    # Compute program ID
    pid_s = tl.program_id(axis=0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)[:,None]
    offs_e = tl.arange(0, e)[None,:]

    tokens_per_expert_ptrs = tokens_per_expert_ptr + offs_e
    logits_softmax_ptrs = logits_softmax_ptr + offs_s * stride_se_s + offs_e
    diag_mask_ptrs = diag_mask_ptr + tl.arange(0, e)[:, None] * stride_se_s + tl.arange(0, e)[None, :]
    
    diag_mask = tl.load(diag_mask_ptrs)
    diag_mask = tl.broadcast_to(tl.expand_dims(diag_mask, axis=0), (BLOCK_S, e, e))

    grad_l_aux_data = tl.load(grad_l_aux_ptr)
    logits_softmax = tl.load(logits_softmax_ptrs)
    tokens_per_expert = tl.load(tokens_per_expert_ptrs)

    grad_me = grad_l_aux_data * ((e * 1e-3) / (s * s * k))
    grad_me_expand = tl.broadcast_to(grad_me, (e))
    grad_softmax = grad_me_expand * tokens_per_expert
    grad_softmax_expand = tl.broadcast_to(grad_softmax, (e, e))
    grad_softmax_expand = tl.expand_dims(grad_softmax_expand, axis=0)
    grad_softmax_expand = tl.broadcast_to(grad_softmax_expand, (BLOCK_S, e, e))
    
    logits_softmax_expand = tl.expand_dims(logits_softmax, axis=2)
    logits_softmax_in1 = tl.broadcast_to(logits_softmax_expand, (BLOCK_S, e, e))
    logits_softmax_in2 = tl.broadcast_to(tl.trans(logits_softmax_expand, 0,2,1), (BLOCK_S, e, e))
    ger = logits_softmax_in1 * logits_softmax_in2
    softmax_grad = diag_mask * logits_softmax_in1 - ger

    grad_logits_data = tl.sum(softmax_grad * grad_softmax_expand, axis=2)
    grad_logits_ptrs = grad_logits_ptr + offs_s * stride_se_s + offs_e
    tl.store(grad_logits_ptrs, grad_logits_data)

def call(tokens_per_expert, logits_softmax, grad_l_aux, k):

    s, e = logits_softmax.shape
    stride_se_s, _ = logits_softmax.stride()

    # Allocate output tensor
    grad_logits = torch.empty((s, e), dtype=logits_softmax.dtype, device=logits_softmax.device)
    diag_mask = torch.diag(torch.ones(e, device=logits_softmax.device))
    grid = lambda META: (triton.cdiv(s, META["BLOCK_S"]), )
    _topk_gating_bwd_kernel[grid](
        tokens_per_expert,
        logits_softmax,
        grad_l_aux,
        diag_mask,
        grad_logits,
        stride_se_s,
        s, 
        e,
        k
    )

    return grad_logits

def bench_fn(tokens_per_expert, logits_softmax, grad_l_aux, k):
    fn = lambda: call(tokens_per_expert, logits_softmax, grad_l_aux, k)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms

# Register operation
name = "_topk_gating_bwd"
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k, c= SymVar('k'), SymVar('c')
        tokens_per_expert = Tensor((experts,), dtype=dtype, device=device)
        logits_softmax = Tensor((seqLen, experts), dtype=dtype, device=device)
        grad_l_aux = Tensor((), dtype=dtype, device=device)
        register_dlblas_op(name, None, (tokens_per_expert, logits_softmax, grad_l_aux, torch.SymInt),
                           call, bench_fn, _topk_gating_bwd_kernel)
