import torch
import triton
import triton.language as tl


def state_merge_torch(o, m, d, other_o, other_m, other_d):
    """
    数值稳定的softmax状态合并
    """
    m_max = torch.maximum(m, other_m)
    d = d * torch.exp2(m - m_max) + other_d * torch.exp2(other_m - m_max)
    o = o * torch.exp2(m - m_max) + other_o * torch.exp2(other_m - m_max)
    return o, m_max, d


def state_normalize_torch(o, m, d):
    """
    归一化状态
    """
    o = o / d
    return o, m, d


def state_get_lse_torch(o, m, d):
    """
    获取log-sum-exp
    """
    return m + torch.log2(d)


@triton.jit
def _merge_normalize_kernel(
    v_a_ptr,
    s_a_ptr,
    v_b_ptr,
    s_b_ptr,
    out_v_ptr,
    out_s_ptr,

    stride_va0,
    stride_va1,
    stride_va2,

    stride_sa0,
    stride_sa1,

    stride_vb0,
    stride_vb1,
    stride_vb2,

    stride_sb0,
    stride_sb1,

    stride_ov0,
    stride_ov1,
    stride_ov2,

    stride_os0,
    stride_os1,

    N_HEADS,
    HEAD_DIM,

    BLOCK_D: tl.constexpr,
):

    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    seq_idx = pid0 // N_HEADS
    head_idx = pid0 % N_HEADS

    offs = pid1 * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = offs < HEAD_DIM

    sa = tl.load(
        s_a_ptr +
        seq_idx * stride_sa0 +
        head_idx * stride_sa1
    )

    sb = tl.load(
        s_b_ptr +
        seq_idx * stride_sb0 +
        head_idx * stride_sb1
    )

    diff = sa - sb

    absd = tl.abs(diff)

    t = tl.exp2(-absd)

    inv = 1.0 / (1.0 + t)

    pred = diff >= 0

    alpha = tl.where(
        pred,
        inv,
        t * inv
    )

    if pid1 == 0:

        s_merged = tl.maximum(sa, sb) + tl.log2(1.0 + t)

        tl.store(
            out_s_ptr +
            seq_idx * stride_os0 +
            head_idx * stride_os1,
            s_merged
        )

    va_ptrs = (
        v_a_ptr +
        seq_idx * stride_va0 +
        head_idx * stride_va1 +
        offs * stride_va2
    )

    vb_ptrs = (
        v_b_ptr +
        seq_idx * stride_vb0 +
        head_idx * stride_vb1 +
        offs * stride_vb2
    )

    out_ptrs = (
        out_v_ptr +
        seq_idx * stride_ov0 +
        head_idx * stride_ov1 +
        offs * stride_ov2
    )

    va = tl.load(
        va_ptrs,
        mask=mask,
        other=0
    )

    vb = tl.load(
        vb_ptrs,
        mask=mask,
        other=0
    )

    va = va.to(tl.float32)
    vb = vb.to(tl.float32)

    out = tl.fma(
        va - vb,
        alpha,
        vb
    )

    tl.store(
        out_ptrs,
        out,
        mask=mask
    )

def _merge_normalize_triton(v_a, s_a, v_b, s_b):
    # Shapes
    seq_len, num_heads, head_dim = v_a.shape

    # Allocate outputs
    out_v = torch.empty((seq_len, num_heads, head_dim), dtype=torch.float32, device=v_a.device)
    out_s = torch.empty((seq_len, num_heads), dtype=torch.float32, device=v_a.device)

    # Launch grid: one program per (seq, head)
    BLOCK_D = 128

    grid = (
        seq_len * num_heads,
        triton.cdiv(head_dim, BLOCK_D)
    )
    _merge_normalize_kernel[grid](
        v_a, s_a, v_b, s_b,
        out_v, out_s,
        v_a.stride(0), v_a.stride(1), v_a.stride(2),
        s_a.stride(0), s_a.stride(1),
        v_b.stride(0), v_b.stride(1), v_b.stride(2),
        s_b.stride(0), s_b.stride(1),
        out_v.stride(0), out_v.stride(1), out_v.stride(2),
        out_s.stride(0), out_s.stride(1),
        num_heads,
        N_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out_v, out_s


def _merge_normalize_fused_torch(v_a, s_a, v_b, s_b):
    """
    Fused, numerically-stable PyTorch implementation that matches the initial semantics.
    Optimized to minimize intermediate tensors and memory traffic on NPU.
    """
    # s_merged via base-2 logaddexp for numerical stability
    s_merged = torch.logaddexp2(s_a, s_b)  # [S, H], fp32

    # Weight for v_a: w_a = 2^(s_a - s_merged) in [0, 1], broadcasting over head_dim
    w_a = (s_a - s_merged).exp2_().unsqueeze(-1)  # [S, H, 1], fp32

    # Merge vectors in fp32 using a single-pass lerp in-place on vb32:
    va32 = v_a.to(torch.float32)
    vb32 = v_b.to(torch.float32)
    vb32.lerp_(va32, w_a)  # vb = vb + (va - vb) * w_a

    return vb32, s_merged


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, v_a, s_a, v_b, s_b):
        """
        Args:
            v_a: [seq_len, num_heads, head_dim] float16
            s_a: [seq_len, num_heads] float32
            v_b: [seq_len, num_heads, head_dim] float16
            s_b: [seq_len, num_heads] float32
        Returns:
            v_merged: [seq_len, num_heads, head_dim] float32
            s_merged: [seq_len, num_heads] float32
        """
        return _merge_normalize_fused_torch(v_a, s_a, v_b, s_b)


# Below are helper utilities for local testing parity with the initial code.
seq_len = 2048
num_heads = 32
head_dim = 128


def get_inputs():
    va = torch.randn(seq_len, num_heads, head_dim, device="npu").half()
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32, device="npu")
    vb = torch.randn(seq_len, num_heads, head_dim, device="npu").half()
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32, device="npu")
    return [va, sa, vb, sb]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = ModelNew(*get_init_inputs()).forward(*get_inputs())
print(out)