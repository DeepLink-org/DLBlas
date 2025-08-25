# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import sys
import os
import torch
import triton
import torch.cuda.nvtx as nvtx
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
try:
    from triton_ops import permute_triton, unpermute_triton
    TRITON_AVAILABLE = True
except ImportError:
    print("Triton ops not found, skipping Triton tests.")
    TRITON_AVAILABLE = False
try:
  from grouped_gemm.ops import permute as permute_topK, unpermute as unpermute_topK
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
print(TRITON_AVAILABLE)
def permute_torch(tokens, indices, expand_factor: int = 1, is_fp8=False):
    """Permute the tokens based on the indices.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token2expert indices tensor.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    expand_factor = indices.size(1)

    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
    return permuted_tokens, sorted_indices


def unpermute_torch(permuted_tokens, sorted_indices, probs: torch.Tensor = None, merge_factor: int = 1):
    """Unpermute the sorted tokens based on the indices.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The sorted indices tensor.
        probs (torch.Tensor, optional): The probabilities tensor. Defaults to None.
        merge_factor (int, optional): The merge factor. Defaults to 1.

    Returns:
        torch.Tensor: The unpermuted tensor.
    """
    merge_factor = probs.size(1)

    if merge_factor > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // merge_factor
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // merge_factor
        assert (
            probs.size(1) == merge_factor
        ), f"probs size {probs.size()} merge_factor {merge_factor}"

    # unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens = permuted_tokens.index_copy(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, merge_factor, permuted_tokens.size(-1))

    if probs is not None:
        dtype = unpermuted_tokens.dtype
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.to(dtype)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


