import torch


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


class Model(torch.nn.Module):
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
        # 执行合并操作
        # 确保输入形状正确
        batch_size, num_heads, head_dim = v_a.shape

        # 扩展标量到与向量相同的形状以便广播
        # [batch_size, num_heads] -> [batch_size, num_heads, 1]
        s_a_expanded = s_a.unsqueeze(-1)
        s_b_expanded = s_b.unsqueeze(-1)

        # 初始化d为1（对应triton中的d=1, other_d=1）
        d_a = torch.ones_like(s_a_expanded)
        d_b = torch.ones_like(s_b_expanded)

        # 执行状态合并
        v_merged, s_max, d = state_merge_torch(
            o=v_a, m=s_a_expanded, d=d_a, other_o=v_b, other_m=s_b_expanded, other_d=d_b
        )

        # 归一化
        v_merged, s_max, d = state_normalize_torch(v_merged, s_max, d)

        # 计算合并后的标量（log-sum-exp）
        s_merged = state_get_lse_torch(v_merged, s_max, d)

        # 去除最后一个维度
        s_merged = s_merged.squeeze(-1)

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
out = Model(*get_init_inputs()).forward(*get_inputs())
print(out)
