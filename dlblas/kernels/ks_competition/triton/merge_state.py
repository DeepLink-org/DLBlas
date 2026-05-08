import torch
import triton
import triton.language as tl


def state_merge_torch(o, m, d, other_o, other_m, other_d):
    """
    数值稳定的softmax状态合并

    Args:
        o, m, d: 第一个状态的输出、最大值和分母
        other_o, other_m, other_d: 第二个状态的输出、最大值和分母

    Returns:
        合并后的输出、最大值和分母
    """
    # 计算两个最大值中的较大者
    m_max = torch.maximum(m, other_m)

    # 数值稳定的指数计算和合并
    # 使用exp2(m - m_max)避免数值溢出
    d = d * torch.exp2(m - m_max) + other_d * torch.exp2(other_m - m_max)
    o = o * torch.exp2(m - m_max) + other_o * torch.exp2(other_m - m_max)

    return o, m_max, d


def state_normalize_torch(o, m, d):
    """
    归一化状态

    Args:
        o: 输出
        m: 最大值
        d: 分母

    Returns:
        归一化后的输出、最大值和分母
    """
    o = o / d
    return o, m, d


def state_get_lse_torch(o, m, d):
    """
    获取log-sum-exp

    Args:
        o: 输出
        m: 最大值
        d: 分母

    Returns:
        log-sum-exp值
    """
    return m + torch.log2(d)


@triton.jit
def _merge_normalize_kernel(
    v_a_ptr,
    s_a_ptr,
    v_b_ptr,
    s_b_ptr,
    v_out_ptr,
    s_out_ptr,
    B,
    H,
    D,
    stride_va0,
    stride_va1,
    stride_va2,
    stride_vb0,
    stride_vb1,
    stride_vb2,
    stride_vo0,
    stride_vo1,
    stride_vo2,
    stride_sa0,
    stride_sa1,
    stride_sb0,
    stride_sb1,
    stride_so0,
    stride_so1,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Offsets for head_dim
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D

    # Compute base ptrs for vectors
    ptr_va = v_a_ptr + pid_b * stride_va0 + pid_h * stride_va1 + offs_d * stride_va2
    ptr_vb = v_b_ptr + pid_b * stride_vb0 + pid_h * stride_vb1 + offs_d * stride_vb2
    ptr_vo = v_out_ptr + pid_b * stride_vo0 + pid_h * stride_vo1 + offs_d * stride_vo2

    # Scalars s_a and s_b for this (b, h)
    ptr_sa = s_a_ptr + pid_b * stride_sa0 + pid_h * stride_sa1
    ptr_sb = s_b_ptr + pid_b * stride_sb0 + pid_h * stride_sb1
    sa = tl.load(ptr_sa).to(tl.float32)
    sb = tl.load(ptr_sb).to(tl.float32)

    # m_max = max(sa, sb)
    m_max = tl.where(sa > sb, sa, sb)

    # exp2 via exp(x * ln2)
    ln2 = 0.6931471805599453
    e_a = tl.exp((sa - m_max) * ln2)
    e_b = tl.exp((sb - m_max) * ln2)
    d = e_a + e_b  # scalar denominator

    # Load vectors, cast to float32
    va = tl.load(ptr_va, mask=mask, other=0).to(tl.float32)
    vb = tl.load(ptr_vb, mask=mask, other=0).to(tl.float32)

    # Merge and normalize
    o = va * e_a + vb * e_b
    v_merged = o * (1.0 / d)
    tl.store(ptr_vo, v_merged, mask=mask)

    # s_merged = m_max + log2(d) = m_max + log(d)/ln2
    s_merged = m_max + tl.log(d) / ln2
    ptr_so = s_out_ptr + pid_b * stride_so0 + pid_h * stride_so1
    tl.store(ptr_so, s_merged)


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, v_a, s_a, v_b, s_b):
        """
        Args:
            v_a_ptr: 第一个状态的向量张量 [batch_size, num_heads, head_dim]
            s_a_ptr: 第一个状态的标量张量 [batch_size, num_heads]
            v_b_ptr: 第二个状态的向量张量 [batch_size, num_heads, head_dim]
            s_b_ptr: 第二个状态的标量张量 [batch_size, num_heads]
        Returns:
            v_merged: 合并后的向量 [batch_size, num_heads, head_dim]
            s_merged: 合并后的标量 [batch_size, num_heads]
        """
        # Input shapes
        batch_size, num_heads, head_dim = v_a.shape

        # Outputs: match PyTorch semantics (float32 for merged vector and scalar)
        v_merged = torch.empty(
            (batch_size, num_heads, head_dim), dtype=torch.float32, device=v_a.device
        )
        s_merged = torch.empty(
            (batch_size, num_heads), dtype=torch.float32, device=s_a.device
        )

        # Strides
        stride_va0, stride_va1, stride_va2 = v_a.stride()
        stride_vb0, stride_vb1, stride_vb2 = v_b.stride()
        stride_vo0, stride_vo1, stride_vo2 = v_merged.stride()
        stride_sa0, stride_sa1 = s_a.stride()
        stride_sb0, stride_sb1 = s_b.stride()
        stride_so0, stride_so1 = s_merged.stride()

        # Grid: one program per (batch, head)
        grid = (batch_size, num_heads)
        BLOCK_SIZE = 128  # tuned for typical head_dim; mask guards OOB
        _merge_normalize_kernel[grid](
            v_a,
            s_a,
            v_b,
            s_b,
            v_merged,
            s_merged,
            batch_size,
            num_heads,
            head_dim,
            stride_va0,
            stride_va1,
            stride_va2,
            stride_vb0,
            stride_vb1,
            stride_vb2,
            stride_vo0,
            stride_vo1,
            stride_vo2,
            stride_sa0,
            stride_sa1,
            stride_sb0,
            stride_sb1,
            stride_so0,
            stride_so1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        return v_merged, s_merged


seq_len = 128
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
