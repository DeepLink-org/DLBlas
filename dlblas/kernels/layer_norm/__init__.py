
import torch
from .layernorm_normal import register as layernorm_normal_register
from .layernorm_opt_2D import register as layernorm_opt_2D_register
from .layernorm_opt_mask_2D_tma import register as layernorm_opt_mask_2D_tma_register
from .layernorm_opt_mask import register as layernorm_opt_mask_register
from .layernorm_opt import register as layernorm_opt_register
from .layernorm_torch import register as layernorm_torch_register
from dlblas.op_registry import op_registry


op_name = 'layernorm'
def call(X, W, B, eps):
    """
    Args:
        X: Input tensor of shape (..., hidden_size)
        W: Weight tensor of shape (hidden_size,)
        B: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability
    Returns:
        Tuple of (output, input, mean, rstd, block_size, num_warps)
    """
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )
    cache_key = f'{op_name}_n_cols={n_cols}'
    op = op_registry.get_op(op_name, (X, W, B, eps), cache_key=cache_key)
    return op(X, W, B, eps)




layernorm_normal_register(op_name)
layernorm_opt_2D_register(op_name)
layernorm_opt_mask_2D_tma_register(op_name)
layernorm_opt_mask_register(op_name)
layernorm_opt_register(op_name)
layernorm_torch_register(op_name)
assert op_registry.get_op_count(name=op_name) == 6