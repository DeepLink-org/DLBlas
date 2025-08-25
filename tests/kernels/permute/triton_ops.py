import torch
import warnings
from dlBLAS.dlblas.kernels.permute.moe_triton_kernels import (
    moe_recover_topk_op_triton, 
    moe_recover_topk_bwd_op_triton,    
   moe_permute_topk_op_triton,
   moe_permute_topk_bwd_op,
)
class PermuteMoE_topK_Triton(torch.autograd.Function):
    workspace = [] 
    @staticmethod
    def forward(ctx,
                input_act: torch.Tensor,
                indices: torch.Tensor):
        if not input_act.numel():
            return input_act, None
        
        if indices.dim() == 1:
            indices = indices.view(-1, 1)

        if not input_act.is_cuda or not indices.is_cuda:
            raise RuntimeError("Triton 内核要求所有输入张量都在 GPU 上。")
        
        if input_act.size(0) != indices.size(0):
            raise RuntimeError(f"输入张量和索引张量的 tokens 数量不匹配: {input_act.size(0)} vs {indices.size(0)}")
        


        num_tokens, num_topK = indices.shape
        num_out_tokens = num_tokens * num_topK
        num_cols=input_act.size(1)  # 确保 input_act 是二维的

        max_expanded_token_num = num_out_tokens
        num_negative_one_in_indices = (indices == -1).sum().item() 
        permuted_output = torch.zeros(num_out_tokens, num_cols, dtype=input_act.dtype, device=input_act.device)
        row_id_map = torch.empty(num_tokens * num_topK, dtype=torch.int32, device='cuda')
        permuted_act, row_id_map, sorted_row_id_result, PermuteMoE_topK_Triton.workspace = moe_permute_topk_op_triton(
            input_act,
            indices,
            num_out_tokens,
            PermuteMoE_topK_Triton.workspace,
            max_expanded_token_num,
            num_negative_one_in_indices,
            permuted_output,
            row_id_map
        )
        ctx.save_for_backward(sorted_row_id_result.clone())
        ctx.original_shape = input_act.shape
        ctx.num_topK = indices.size(1)
        return permuted_act, row_id_map

    @staticmethod
    def backward(ctx, permuted_act_grad, _):
        if not permuted_act_grad.numel():
            return permuted_act_grad, None

        if not permuted_act_grad.is_contiguous():
            permuted_act_grad = permuted_act_grad.contiguous()

        sorted_row_id, = ctx.saved_tensors

        unpermuted_grad = torch.zeros(ctx.original_shape, dtype=torch.float32, device=permuted_act_grad.device)

        unpermuted_grad = moe_permute_topk_bwd_op(
            permuted_act_grad,
            sorted_row_id,
            ctx.original_shape,
            ctx.num_topK
        )
        return unpermuted_grad, None




class UnpermuteMoE_topK_Triton(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,
                input_act: torch.Tensor,
                row_id_map: torch.Tensor,
                probs: torch.Tensor):
        if not input_act.numel():
            ctx.probs = probs
            return input_act
        
        if not input_act.is_cuda or not row_id_map.is_cuda or (probs is not None and not probs.is_cuda):
            raise RuntimeError("Triton 内核要求所有输入张量都在 GPU 上。")
            
        if not input_act.is_contiguous(): input_act = input_act.contiguous()
        if not row_id_map.is_contiguous(): row_id_map = row_id_map.contiguous()
        if probs is not None and not probs.is_contiguous(): probs = probs.contiguous()

        num_tokens = probs.size(0) if probs is not None else row_id_map.size(1)
        num_topK = probs.size(1) if probs is not None else row_id_map.size(0)

        unpermuted_output = moe_recover_topk_op_triton(
            input_act,
            row_id_map,
            probs,
            num_tokens,
            num_topK
        )
        ctx.save_for_backward(input_act, row_id_map, probs)
        return unpermuted_output


    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.probs
        if not unpermuted_act_grad.is_contiguous():
            unpermuted_act_grad = unpermuted_act_grad.contiguous()
        input_act, row_id_map, probs = ctx.saved_tensors
        act_grad_fp32, prob_grad = moe_recover_topk_bwd_op_triton(
            unpermuted_act_grad,
            input_act,
            row_id_map,
            probs
        )
        act_grad = act_grad_fp32.to(input_act.dtype)

        if not ctx.needs_input_grad[2]: 
            prob_grad = None
        
        return act_grad, None, prob_grad


def permute_triton(input_act, indices):
  """
  使用 Triton 内核对 MoE 的 token 进行重排。
  
  Args:
      input_act (torch.Tensor): 输入的激活值，形状为 [num_tokens, hidden_size]。
      indices (torch.Tensor): 路由索引，形状为 [num_tokens, topK]。
  
  Returns:
      Tuple[torch.Tensor, torch.Tensor]: (重排后的激活值, 行ID映射图)
  """
  return PermuteMoE_topK_Triton.apply(input_act, indices)


def unpermute_triton(input_act, row_id_map, probs):
  """
  使用 Triton 内核恢复 MoE 的 token 顺序并进行加权求和。
  
  Args:
      input_act (torch.Tensor): 经过专家网络处理后的激活值。
      row_id_map (torch.Tensor): permute 操作生成的行ID映射图。
      probs (torch.Tensor): 门控网络给出的权重，形状为 [num_tokens, topK]。
  
  Returns:
      torch.Tensor: 恢复顺序并加权求和后的最终激活值。
  """
  return UnpermuteMoE_topK_Triton.apply(input_act, row_id_map, probs)

