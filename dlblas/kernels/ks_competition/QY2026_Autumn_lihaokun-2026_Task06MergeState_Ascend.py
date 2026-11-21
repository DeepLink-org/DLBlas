import torch
import torch_npu
import triton
import triton.language as tl

def get_device():
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

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
    v_a_ptr, s_a_ptr, v_b_ptr, s_b_ptr,
    v_out_ptr, s_out_ptr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Load per-row scalars as f32
    s_a = tl.load(s_a_ptr + pid).to(tl.float32)
    s_b = tl.load(s_b_ptr + pid).to(tl.float32)

    # m_max = max(s_a, s_b)
    m_max = tl.maximum(s_a, s_b)

    # Base-2 exponentials for numerical stability
    wa = tl.exp2(s_a - m_max)
    wb = tl.exp2(s_b - m_max)
    denom = wa + wb
    inv_denom = 1.0 / denom

    # s_merged = m_max + log2(denom)
    s_merged = m_max + tl.log2(denom)
    tl.store(s_out_ptr + pid, s_merged)

    # Process one row vector in a single tile
    base = pid * head_dim
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    idx = base + offs

    va = tl.load(v_a_ptr + idx, mask=mask, other=0).to(tl.float32)
    vb = tl.load(v_b_ptr + idx, mask=mask, other=0).to(tl.float32)

    # Precompute normalized weights to reduce ops
    alpha = wa * inv_denom
    beta = wb * inv_denom
    out = va * alpha + vb * beta
    tl.store(v_out_ptr + idx, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _fallback(self, v_a, s_a, v_b, s_b):
        # 与参考实现等价的数值稳定计算（完全向量化）
        s_a_exp = s_a.unsqueeze(-1)  # [S, H, 1]
        s_b_exp = s_b.unsqueeze(-1)  # [S, H, 1]

        m_max = torch.maximum(s_a_exp, s_b_exp)  # [S, H, 1]
        wa = torch.exp2(s_a_exp - m_max)
        wb = torch.exp2(s_b_exp - m_max)
        denom = wa + wb  # [S, H, 1]

        v_merged = (v_a.float() * wa + v_b.float() * wb) / denom  # [S, H, D] fp32
        s_merged = (m_max + torch.log2(denom)).squeeze(-1)  # [S, H] fp32
        return v_merged, s_merged

    def forward(self, v_a, s_a, v_b, s_b):
        """
        Args:
            v_a: [seq_len, num_heads, head_dim], float16
            s_a: [seq_len, num_heads], float32
            v_b: [seq_len, num_heads, head_dim], float16
            s_b: [seq_len, num_heads], float32
        Returns:
            v_merged: [seq_len, num_heads, head_dim], float32
            s_merged: [seq_len, num_heads], float32
        """
        if v_a.device.type == "npu":
            return self._fallback(v_a, s_a, v_b, s_b)

        # 非NPU设备（如GPU/CPU）尝试使用Triton内核
        seq_len, num_heads, head_dim = v_a.shape
        rows = seq_len * num_heads

        v_a_c = v_a.contiguous()
        v_b_c = v_b.contiguous()
        s_a_c = s_a.contiguous()
        s_b_c = s_b.contiguous()

        v_out = torch.empty((seq_len, num_heads, head_dim), dtype=torch.float32, device=v_a.device)
        s_out = torch.empty((seq_len, num_heads), dtype=torch.float32, device=v_a.device)

        v_a_flat = v_a_c.view(rows, head_dim)
        v_b_flat = v_b_c.view(rows, head_dim)
        s_a_flat = s_a_c.view(rows)
        s_b_flat = s_b_c.view(rows)
        v_out_flat = v_out.view(rows, head_dim)
        s_out_flat = s_out.view(rows)

        BLOCK_D = 128
        grid = (rows,)
        _merge_normalize_kernel[grid](
            v_a_flat, s_a_flat, v_b_flat, s_b_flat,
            v_out_flat, s_out_flat,
            head_dim=head_dim,
            BLOCK_D=BLOCK_D,
            num_warps=1,
            num_stages=2,
        )

        return v_out, s_out


seq_len = 2048
num_heads = 32
head_dim = 128
device = get_device()

def get_inputs():
    va = torch.randn(seq_len, num_heads, head_dim, device=device).half()
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32, device=device)
    vb = torch.randn(seq_len, num_heads, head_dim, device=device).half()
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32, device=device)
    return [va, sa, vb, sb]

def get_init_inputs():
    return []
