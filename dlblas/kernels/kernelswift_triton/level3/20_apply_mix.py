import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mix_sum_kernel(
    x_ptr, mix_ptr, y_ptr,
    stride_x_n0, stride_x_n1, stride_x_mhc, stride_x_h,
    stride_m_n0, stride_m_n1, stride_m_mhc, stride_m_h,  # kept for signature consistency
    stride_y_n0, stride_y_n1, stride_y_h,
    N0, N1, H,
    M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    n1_idx = pid_row % N1
    n0_idx = pid_row // N1

    h_start = pid_col * BLOCK_H
    offs_h = h_start + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    full_tile = (h_start + BLOCK_H) <= H  # uniform within program instance

    # Base offsets
    base_x = n0_idx * stride_x_n0 + n1_idx * stride_x_n1 + offs_h * stride_x_h
    base_m = n0_idx * stride_m_n0 + n1_idx * stride_m_n1
    base_y = n0_idx * stride_y_n0 + n1_idx * stride_y_n1 + offs_h * stride_y_h

    # Accumulator in fp32 for correctness (bf16 * f32 -> f32)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    if M == 4:
        # Preload mix scalars
        m0 = tl.load(mix_ptr + base_m + 0 * stride_m_mhc).to(tl.float32)
        m1 = tl.load(mix_ptr + base_m + 1 * stride_m_mhc).to(tl.float32)
        m2 = tl.load(mix_ptr + base_m + 2 * stride_m_mhc).to(tl.float32)
        m3 = tl.load(mix_ptr + base_m + 3 * stride_m_mhc).to(tl.float32)

        x_base = x_ptr + base_x
        if full_tile:
            x0 = tl.load(x_base + 0 * stride_x_mhc).to(tl.float32)
            x1 = tl.load(x_base + 1 * stride_x_mhc).to(tl.float32)
            x2 = tl.load(x_base + 2 * stride_x_mhc).to(tl.float32)
            x3 = tl.load(x_base + 3 * stride_x_mhc).to(tl.float32)
        else:
            x0 = tl.load(x_base + 0 * stride_x_mhc, mask=mask_h, other=0.0).to(tl.float32)
            x1 = tl.load(x_base + 1 * stride_x_mhc, mask=mask_h, other=0.0).to(tl.float32)
            x2 = tl.load(x_base + 2 * stride_x_mhc, mask=mask_h, other=0.0).to(tl.float32)
            x3 = tl.load(x_base + 3 * stride_x_mhc, mask=mask_h, other=0.0).to(tl.float32)

        # Increase ILP via pairwise accumulation
        acc = (x0 * m0 + x2 * m2) + (x1 * m1 + x3 * m3)
    elif M <= 8:
        # Small-M generic unrolled path
        # Preload all mix scalars
        m_vals = tl.zeros([M], dtype=tl.float32)
        m_ptr = mix_ptr + base_m
        for k in tl.static_range(0, M):
            m_vals[k] = tl.load(m_ptr + k * stride_m_mhc).to(tl.float32)

        x_base = x_ptr + base_x
        # Unrolled accumulation
        for k in tl.static_range(0, M):
            if full_tile:
                xk = tl.load(x_base + k * stride_x_mhc).to(tl.float32)
            else:
                xk = tl.load(x_base + k * stride_x_mhc, mask=mask_h, other=0.0).to(tl.float32)
            acc += xk * m_vals[k]
    else:
        # Generic pipelined path for larger M
        x_ptr_mh = x_ptr + base_x
        m_ptr_mh = mix_ptr + base_m
        if full_tile:
            x_curr = tl.load(x_ptr_mh).to(tl.float32)
        else:
            x_curr = tl.load(x_ptr_mh, mask=mask_h, other=0.0).to(tl.float32)
        m_curr = tl.load(m_ptr_mh).to(tl.float32)

        for _ in tl.static_range(1, M):
            if full_tile:
                x_next = tl.load(x_ptr_mh + stride_x_mhc).to(tl.float32)
            else:
                x_next = tl.load(x_ptr_mh + stride_x_mhc, mask=mask_h, other=0.0).to(tl.float32)
            m_next = tl.load(m_ptr_mh + stride_m_mhc).to(tl.float32)

            acc += x_curr * m_curr

            x_ptr_mh += stride_x_mhc
            m_ptr_mh += stride_m_mhc
            x_curr = x_next
            m_curr = m_next

        acc += x_curr * m_curr

    # Store result as bf16
    tl.store(y_ptr + base_y, acc.to(tl.bfloat16), mask=mask_h)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        # Fallback for non-CUDA tensors
        if not x.is_cuda or not mix.is_cuda:
            return (x * mix).sum(-2).bfloat16()

        assert x.dim() == 4 and mix.dim() == 4, "Expected x: [n0, n1, mhc, h], mix: [n0, n1, mhc, 1]"
        n0, n1, mhc, h = x.shape
        y = torch.empty((n0, n1, h), dtype=torch.bfloat16, device=x.device)

        sx_n0, sx_n1, sx_mhc, sx_h = x.stride()
        sm_n0, sm_n1, sm_mhc, sm_h = mix.stride()
        sy_n0, sy_n1, sy_h = y.stride()

        # Tuned config for H200
        BLOCK_H = 256
        grid = (n0 * n1, triton.cdiv(h, BLOCK_H))

        mix_sum_kernel[grid](
            x, mix, y,
            sx_n0, sx_n1, sx_mhc, sx_h,
            sm_n0, sm_n1, sm_mhc, sm_h,
            sy_n0, sy_n1, sy_h,
            n0, n1, h,
            M=mhc,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
        return y


n0=2
n1=1024
mhc=4
h=1280

def generate_pre_apply_mix_test_data(
    n0: int, n1: int, mhc: int, h: int
) -> dict[str, torch.Tensor]:
    x = torch.randn(n0, n1, mhc, h, dtype=torch.bfloat16, device="cuda").sigmoid()
    mix = torch.randn(n0, n1, mhc, 1, dtype=torch.float32, device="cuda").softmax(-2)
    o_grad = torch.randn(n0, n1, h, dtype=torch.bfloat16, device="cuda")

    return [x,mix,o_grad]

def get_inputs():
    x,mix,o_grad = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc, h=h)
    return [x,mix]

def get_init_inputs():
    return []
