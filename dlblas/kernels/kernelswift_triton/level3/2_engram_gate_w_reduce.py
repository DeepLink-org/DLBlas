import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_reduce_update_kernel(
    grad_w_partial_ptr,  # float32 [B, M, N]
    weight_hidden_ptr,   # bfloat16 [M, N]
    weight_embed_ptr,    # bfloat16 [M, N]
    grad_weight_hidden_ptr,  # float32 [M, N]
    grad_weight_embed_ptr,   # float32 [M, N]
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    stride_gwp_b: tl.int32,
    stride_gwp_m: tl.int32,
    stride_gwp_n: tl.int32,
    stride_wh_m: tl.int32,
    stride_wh_n: tl.int32,
    stride_we_m: tl.int32,
    stride_we_n: tl.int32,
    stride_gwh_m: tl.int32,
    stride_gwh_n: tl.int32,
    stride_gwe_m: tl.int32,
    stride_gwe_n: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_broadcast = offs_m[:, None]
    n_broadcast = offs_n[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer for the (M, N) tile; advance along B with pointer increments
    base_ptr = grad_w_partial_ptr + m_broadcast * stride_gwp_m + n_broadcast * stride_gwp_n
    ptr = base_ptr
    b = 0
    # Unroll the reduction over B for better ILP
    while b + 3 < B:
        acc += tl.load(ptr, mask=mask, other=0.0)
        ptr = ptr + stride_gwp_b
        acc += tl.load(ptr, mask=mask, other=0.0)
        ptr = ptr + stride_gwp_b
        acc += tl.load(ptr, mask=mask, other=0.0)
        ptr = ptr + stride_gwp_b
        acc += tl.load(ptr, mask=mask, other=0.0)
        ptr = ptr + stride_gwp_b
        b += 4
    while b < B:
        acc += tl.load(ptr, mask=mask, other=0.0)
        ptr = ptr + stride_gwp_b
        b += 1

    we_ptrs = weight_embed_ptr + m_broadcast * stride_we_m + n_broadcast * stride_we_n
    wh_ptrs = weight_hidden_ptr + m_broadcast * stride_wh_m + n_broadcast * stride_wh_n

    we = tl.load(we_ptrs, mask=mask, other=0).to(tl.float32)
    wh = tl.load(wh_ptrs, mask=mask, other=0).to(tl.float32)

    gwh_ptrs = grad_weight_hidden_ptr + m_broadcast * stride_gwh_m + n_broadcast * stride_gwh_n
    gwe_ptrs = grad_weight_embed_ptr + m_broadcast * stride_gwe_m + n_broadcast * stride_gwe_n

    gwh = tl.load(gwh_ptrs, mask=mask, other=0.0)
    gwe = tl.load(gwe_ptrs, mask=mask, other=0.0)

    gwh = gwh + acc * we
    gwe = gwe + acc * wh

    tl.store(gwh_ptrs, gwh, mask=mask)
    tl.store(gwe_ptrs, gwe, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed):
        # Fallback to PyTorch on CPU or if Triton is unavailable
        use_triton = (
            grad_w_partial.is_cuda
            and weight_hidden.is_cuda
            and weight_embed.is_cuda
            and grad_weight_hidden.is_cuda
            and grad_weight_embed.is_cuda
        )
        if not use_triton:
            grad_w_sum = grad_w_partial.sum(0)
            grad_weight_hidden += grad_w_sum * weight_embed.float()
            grad_weight_embed += grad_w_sum * weight_hidden.float()
            return grad_weight_hidden, grad_weight_embed

        B, M, N = grad_w_partial.shape

        stride_gwp_b, stride_gwp_m, stride_gwp_n = grad_w_partial.stride()
        stride_wh_m, stride_wh_n = weight_hidden.stride()
        stride_we_m, stride_we_n = weight_embed.stride()
        stride_gwh_m, stride_gwh_n = grad_weight_hidden.stride()
        stride_gwe_m, stride_gwe_n = grad_weight_embed.stride()

        BLOCK_M = 1
        BLOCK_N = 256
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))

        _fused_reduce_update_kernel[grid](
            grad_w_partial,
            weight_hidden,
            weight_embed,
            grad_weight_hidden,
            grad_weight_embed,
            B, M, N,
            stride_gwp_b, stride_gwp_m, stride_gwp_n,
            stride_wh_m, stride_wh_n,
            stride_we_m, stride_we_n,
            stride_gwh_m, stride_gwh_n,
            stride_gwe_m, stride_gwe_n,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return grad_weight_hidden, grad_weight_embed


def generate_test_data(hidden_size):
    hc_mult = 4
    num_persistent_blocks = 108
    grad_w_partial = torch.randn(num_persistent_blocks, hc_mult, hidden_size, dtype=torch.float32)
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device='cpu')
    return (grad_w_partial, weight_hidden, weight_embed)

def test_engram_grad_w_reduce():
    return Model(*get_init_inputs()).forward(*get_inputs())

def get_inputs():
    hidden_size = 4096
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(hidden_size)
    hc_mult = grad_w_partial.shape[1]
    grad_wh_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cpu')
    grad_we_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device='cpu')
    return [grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref]

def get_init_inputs():
    return []