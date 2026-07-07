# ----------------------------------------------------------------------------------------------------------
# engram_gate_w_reduce Golden 计算（双通路共用）
#
# grad_w_sum = sum(grad_w_partial, dim=0)           # [108, 4, H] → [4, H]
# grad_weight_hidden += grad_w_sum * weight_embed   # broadcast mul-add
# grad_weight_embed += grad_w_sum * weight_hidden   # broadcast mul-add
# ----------------------------------------------------------------------------------------------------------

import numpy as np


def bf16_to_fp32(arr_bf16):
    """将 bfloat16 (uint16表示) 转换为 float32"""
    arr_u32 = arr_bf16.astype(np.uint32) << 16
    return arr_u32.view(np.float32)


def compute_golden(grad_w_partial, weight_hidden, weight_embed,
                   grad_weight_hidden_in, grad_weight_embed_in):
    """计算 engram_gate_w_reduce 的参考输出 (FP32 全精度).

    Args:
        grad_w_partial:       [108, 4, hidden_size] float32 numpy array
        weight_hidden:        [4, hidden_size] float32 numpy array (for golden; kernel uses BF16)
        weight_embed:         [4, hidden_size] float32 numpy array
        grad_weight_hidden_in: [4, hidden_size] float32 numpy array (初始值)
        grad_weight_embed_in:  [4, hidden_size] float32 numpy array (初始值)

    Returns:
        grad_weight_hidden_out: [4, hidden_size] float32
        grad_weight_embed_out:  [4, hidden_size] float32
    """
    # 先转换 BF16→FP32 以模拟 kernel 精度（如果有 BF16 输入）
    # 这里 weight_hidden/weight_embed 已经是 FP32，需要先转回 BF16 再转 FP32
    # 以模拟 kernel 中的 Cast 操作
    def sim_bf16_cast(arr_fp32):
        arr_u32 = arr_fp32.view(np.uint32)
        arr_bf16 = (arr_u32 >> 16).astype(np.uint16)
        arr_u32_back = arr_bf16.astype(np.uint32) << 16
        return arr_u32_back.view(np.float32)

    # 为更精确的比对，在 golden 中模拟 BF16 转换
    wh_fp32 = sim_bf16_cast(weight_hidden)
    we_fp32 = sim_bf16_cast(weight_embed)

    # Step 1: Reduction
    grad_w_sum = np.sum(grad_w_partial, axis=0)  # [4, H]

    # Step 2: Multiply-Accumulate
    grad_weight_hidden_out = grad_weight_hidden_in.copy()
    grad_weight_hidden_out += grad_w_sum * we_fp32

    grad_weight_embed_out = grad_weight_embed_in.copy()
    grad_weight_embed_out += grad_w_sum * wh_fp32

    return grad_weight_hidden_out, grad_weight_embed_out
