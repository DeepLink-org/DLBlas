import pytest
import torch
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.layer_norm.layernorm_normal import call as layernorm_normal
from dlblas.kernels.layer_norm.layernorm_normal_loop import call as layernorm_normal_loop
from dlblas.kernels.layer_norm.layernorm_opt_2D import call as layernorm_opt_2D
from dlblas.kernels.layer_norm.layernorm_opt_mask_2D_tma import call as layernorm_opt_mask_2D_tma
from dlblas.kernels.layer_norm.layernorm_opt_mask import call as layernorm_opt_mask
from dlblas.kernels.layer_norm.layernorm_opt import call as layernorm_opt
from dlblas.kernels.layer_norm.layernorm_torch import call as layernorm_torch

device = infer_device()

@pytest.mark.parametrize("triton_op", [layernorm_normal, layernorm_normal_loop, 
layernorm_opt_2D, layernorm_opt_mask_2D_tma, layernorm_opt_mask, layernorm_opt, layernorm_torch])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [32*1024])
@pytest.mark.parametrize("hidden_size", [256, 4096, 16 * 1024, 17*1024, 64 * 1024])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [(torch.bfloat16, 2e-2, 2e-2),],
)
def test_layer_norm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    triton_op,
) -> None:
    if triton_op == layernorm_normal_loop and hidden_size >= 32*1024:
        return
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    triton_x = x.clone().requires_grad_(False)
    torch_x = x.clone().requires_grad_(False)
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    bias = torch.randn(hidden_size, dtype=dtype, device=device)
    triton_output = triton_op(triton_x, weight, bias, eps=1e-6)
    torch_output = torch.layer_norm(torch_x, (hidden_size,), weight, bias, eps=1e-6)
    assert torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol)