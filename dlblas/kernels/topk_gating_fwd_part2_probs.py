import torch
import triton
import triton.language as tl
import triton.language.core as tlc
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace
from dlblas.utils.libentry import libentry
# from tutel import moe as tutel_moe

@libentry()
@triton.jit
def _topk_gating_fwd_part2_probs(
    gates,
    masks,
    masks_gates,
    masks_with_capacity,
    tokens_per_expert_before_capacity,
    aggregated_probs_per_expert_ptr,
    aux_loss_per_expert_ptr,
    fill_value,
    stride_s,
    moe_aux_loss_coeff,
    SEQ_LEN: tl.constexpr, 
    BLOCK_S: tl.constexpr, 
    K: tl.constexpr,
    EXPERTS: tl.constexpr,
    KS: tl.constexpr,
    BLOCK_KS: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    pid_e = tl.program_id(axis=0)
    offs_g = tl.arange(0, BLOCK_S)
    masks_ptrs = masks + offs_g * stride_s + pid_e
    mask_data = tl.load(masks_ptrs, mask=offs_g < SEQ_LEN)
    masks_gates_ptrs = masks_gates + offs_g * stride_s + pid_e
    mask_gates_data = tl.load(masks_gates_ptrs, mask=offs_g < SEQ_LEN)

    scores_ptrs = gates + offs_g * stride_s + pid_e
    scores_data = tl.load(scores_ptrs, mask=offs_g < SEQ_LEN)
    aggregated_probs_per_expert = tl.sum(scores_data, axis = 0)
    aggregated_probs_per_expert_ptrs = aggregated_probs_per_expert_ptr + pid_e
    tl.store(aggregated_probs_per_expert_ptrs, aggregated_probs_per_expert)

    tokens_per_expert_before_capacity_data = tl.sum(mask_data, axis = 0)
    tokens_per_expert_before_capacity_ptr = tokens_per_expert_before_capacity + pid_e
    tl.store(tokens_per_expert_before_capacity_ptr, tokens_per_expert_before_capacity_data)

    aux_loss_per_expert = aggregated_probs_per_expert * tokens_per_expert_before_capacity_data * EXPERTS * moe_aux_loss_coeff / (SEQ_LEN * SEQ_LEN * K)
    aux_loss_per_expert_ptrs = aux_loss_per_expert_ptr + pid_e
    tl.store(aux_loss_per_expert_ptrs, aux_loss_per_expert)

    masks_with_capacity_data = tl.zeros((SEQ_LEN, ), dtype=tl.int64)
    for idx in tlc.static_range(CAPACITY):
        max_idx = tl.argmax(mask_gates_data, axis = 0, tie_break_left=False)
        all_ids = tl.arange(0, SEQ_LEN)
        max_idx_expand = tl.broadcast_to(max_idx, (SEQ_LEN))
        mask_with_capacity_data = tl.where(all_ids == max_idx_expand, 1, 0)
        masks_with_capacity_data += mask_with_capacity_data

        mask_gates_data = tl.where(mask_with_capacity_data > 0, fill_value, mask_gates_data)
    
    masks_with_capacity_data = masks_with_capacity_data * mask_data

    masks_with_capacity_ptrs = masks_with_capacity + offs_g * stride_s + pid_e
    tl.store(masks_with_capacity_ptrs, masks_with_capacity_data, mask=offs_g < SEQ_LEN)


def call(gates: torch.Tensor, masks: torch.Tensor, masks_gates: torch.Tensor, k: int, capacity: int, moe_aux_loss_coeff: float):
    s, e = gates.shape
    stride_se_s, _ = gates.stride()
    # print("_topk_gating_fwd_part2_probs  topk_masks:\n", masks)
    # topk_mask_str = masks.detach().cpu().numpy()  # 确保张量在CPU上，并且转换为NumPy数组
    # topk_mask_str = str(topk_mask_str)

    # # 写入文本文件
    # with open("/home/aigc/PRJ/Triton/python/dlBLAS/benchmarks/test_gating/megatron_topk_mask.txt", "a") as f:
    #     f.write(topk_mask_str + "\n")
    masks_with_capacity = torch.empty((s, e), dtype=torch.float32, device=gates.device)
    tokens_per_expert_before_capacity = torch.empty((e), dtype=torch.float32, device=gates.device)
    aggregated_probs_per_expert = torch.empty((e), dtype=gates.dtype, device=gates.device)
    aux_loss_per_expert = torch.empty((e), dtype=gates.dtype, device=gates.device)
    fill_value = torch.finfo(gates.dtype).min
    with torch.cuda.device(gates.device):
        _topk_gating_fwd_part2_probs[(e,)](
            gates,
            masks,
            masks_gates,
            masks_with_capacity,  
            tokens_per_expert_before_capacity,
            aggregated_probs_per_expert,
            aux_loss_per_expert,
            fill_value,
            stride_se_s,
            moe_aux_loss_coeff,
            SEQ_LEN = s,
            BLOCK_S= triton.next_power_of_2(s),
            K = k,
            EXPERTS = e,
            KS = k * s,
            BLOCK_KS = triton.next_power_of_2(k * s),
            CAPACITY = capacity,
        )    
    return masks_with_capacity, tokens_per_expert_before_capacity, aggregated_probs_per_expert, aux_loss_per_expert


def bench_fn(gates: torch.Tensor, masks: torch.Tensor, masks_gates: torch.Tensor, k: int, capacity: int, moe_aux_loss_coeff: float):
    fn = lambda: call(gates, masks, masks_gates, k, capacity, moe_aux_loss_coeff)
    ms = triton.testing.do_bench(fn, warmup=20, rep=20)
    return ms


# register
name = '_topk_gating_fwd_part2_probs'
for dtype in [torch.float16, torch.float32]:
    for device in ['cuda']:
        seqLen, experts = SymVar('seqLen'), SymVar('experts')
        k = SymVar('k')
        capacity = SymVar('capacity')
        moe_aux_loss_coeff = SymVar('moe_aux_loss_coeff')
        # we dont' actually allocate tensor
        logits = Tensor((seqLen, experts), dtype=dtype, device=device)
        masks = Tensor((seqLen, experts), dtype=torch.int64, device=device)
        masks_gates = Tensor((seqLen, experts), dtype=dtype, device=device)
        register_dlblas_op(name, None, (logits, masks, masks_gates, torch.SymInt, torch.SymInt, torch.SymFloat), call, bench_fn, _topk_gating_fwd_part2_probs)


def main():
    # 设置随机种子
    torch.manual_seed(0)

    # 定义 gates 和 masks 的维度
    s, e, k = 16, 8, 4  # 示例：序列长度 16，专家数 8，TopK 选择 4

    # 创建输入张量
    gates = torch.randn((s, e), device='cuda', dtype=torch.float32)  # gates 张量
    masks = torch.randint(0, 1, (s, e), device='cuda', dtype=torch.int64)  # masks 张量

    # gates = torch.ones((s, e), device='cuda', dtype=torch.float32)  # gates tensor, all 1
    # masks = torch.ones((k, s, e), device='cuda', dtype=torch.int64)  # masks tensor, all 0


    # 使用 call 函数进行计算
    locations, res, ce = call(gates, masks, k)

    # 打印输出结果
    # print("Locations:", locations)
    # print("Result:", res)
    # print("CE:", ce)

if __name__ == "__main__":
    main()