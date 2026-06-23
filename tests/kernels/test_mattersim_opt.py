# Copyright (c) 2025, DeepLink.
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Triton tests"
)


@pytest.fixture(scope="module")
def kernels():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    module_path = (
        Path(__file__).resolve().parents[2] / "dlblas" / "kernels" / "mattersim_opt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mattersim_opt_test_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_activation(x, activation):
    if activation == "none" or activation is None:
        return x
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation in ("swish", "silu"):
        return F.silu(x)
    raise AssertionError(f"bad activation {activation}")


@pytest.mark.parametrize("activation", ["none", "sigmoid", "swish"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_bias_activation(kernels, activation, use_bias):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(17, 31, device=device, dtype=torch.float32) * 0.1
    weight = torch.randn(29, 31, device=device, dtype=torch.float32) * 0.1
    bias = (
        torch.randn(29, device=device, dtype=torch.float32) * 0.1 if use_bias else None
    )

    out = kernels.linear_bias_activation(x, weight, bias, activation=activation)
    ref = _apply_activation(F.linear(x, weight, bias), activation)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


def test_addmm_sigmoid_mul_outputs(kernels):
    torch.manual_seed(1)
    device = "cuda"
    x = torch.randn(23, 19, device=device, dtype=torch.float32) * 0.1
    weight = torch.randn(37, 19, device=device, dtype=torch.float32) * 0.1
    bias = torch.randn(37, device=device, dtype=torch.float32) * 0.1

    preact, sigmoid, swish = kernels.addmm_sigmoid_mul(x, weight, bias)
    ref_preact = F.linear(x, weight, bias)
    ref_sigmoid = torch.sigmoid(ref_preact)
    torch.testing.assert_close(preact, ref_preact, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(sigmoid, ref_sigmoid, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(swish, ref_preact * ref_sigmoid, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        kernels.addmm_sigmoid(x, weight, bias), ref_sigmoid, rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_two_path_two_layer_mlp(kernels, dtype):
    torch.manual_seed(2)
    device = "cuda"
    x = torch.randn(21, 17, device=device, dtype=dtype) * 0.1
    w1g = torch.randn(32, 17, device=device, dtype=dtype) * 0.1
    b1g = torch.randn(32, device=device, dtype=dtype) * 0.1
    w2g = torch.randn(25, 32, device=device, dtype=dtype) * 0.1
    b2g = torch.randn(25, device=device, dtype=dtype) * 0.1
    w1s = torch.randn(32, 17, device=device, dtype=dtype) * 0.1
    b1s = torch.randn(32, device=device, dtype=dtype) * 0.1
    w2s = torch.randn(25, 32, device=device, dtype=dtype) * 0.1
    b2s = torch.randn(25, device=device, dtype=dtype) * 0.1

    out = kernels.two_path_two_layer_mlp(x, w1g, b1g, w2g, b2g, w1s, b1s, w2s, b2s)
    g = F.silu(F.linear(x, w1g, b1g))
    g = F.silu(F.linear(g, w2g, b2g))
    s = F.silu(F.linear(x, w1s, b1s))
    s = torch.sigmoid(F.linear(s, w2s, b2s))
    tol = 2e-3 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(out, g * s, rtol=tol, atol=tol)


def test_fused_elementwise_backward_helpers(kernels):
    torch.manual_seed(3)
    device = "cuda"
    shape = (257,)
    t1 = torch.randn(shape, device=device)
    a1 = torch.randn(shape, device=device)
    s1 = torch.sigmoid(torch.randn(shape, device=device))
    s3 = torch.sigmoid(torch.randn(shape, device=device))
    mul5, sbwd = kernels.fused_mul5_sigmoid_bwd(t1, a1, s1, s3)
    torch.testing.assert_close(mul5, t1 * s3)
    torch.testing.assert_close(
        sbwd, t1 * (a1 * s1) * s3 * (1 - s3), rtol=1e-5, atol=1e-5
    )

    x = torch.randn(shape, device=device)
    addmm = torch.randn(shape, device=device)
    sigmoid = torch.sigmoid(torch.randn(shape, device=device))
    out = kernels.fused_add_swish_bwd(x, addmm, sigmoid)
    torch.testing.assert_close(out, x * (sigmoid + addmm * sigmoid * (1 - sigmoid)))


def test_scatter_add_2d_dim0(kernels):
    torch.manual_seed(4)
    device = "cuda"
    src = torch.randn(113, 65, device=device, dtype=torch.float32)
    index = torch.randint(0, 17, (113,), device=device, dtype=torch.long)
    out = kernels.scatter_add_2d_dim0(src, index, dim_size=19)
    ref = torch.zeros(19, 65, device=device, dtype=torch.float32)
    ref.scatter_add_(0, index.view(-1, 1).expand_as(src), src)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_offset_three_body_indices(kernels):
    device = "cuda"
    three_body = torch.tensor(
        [[0, 1], [1, 2], [0, 1], [2, 3], [3, 4]],
        device=device,
        dtype=torch.long,
    )
    num_three_body = torch.tensor([2, 3], device=device, dtype=torch.long)
    num_bonds = torch.tensor([3, 5], device=device, dtype=torch.long)
    out = kernels.offset_three_body_indices(three_body, num_three_body, num_bonds)
    ref = torch.tensor(
        [[0, 1], [1, 2], [3, 4], [5, 6], [6, 7]],
        device=device,
        dtype=torch.long,
    )
    torch.testing.assert_close(out, ref)


def test_fill_threebody_indices_grouped_edges(kernels):
    device = "cuda"
    orig_idx = torch.tensor([10, 11, 12, 20, 21], device=device, dtype=torch.long)
    k_per_atom = torch.tensor([3, 2], device=device, dtype=torch.long)
    out = kernels.fill_threebody_indices(orig_idx, k_per_atom)
    ref = torch.tensor(
        [
            [10, 11],
            [10, 12],
            [11, 10],
            [11, 12],
            [12, 10],
            [12, 11],
            [20, 21],
            [21, 20],
        ],
        device=device,
        dtype=torch.long,
    )
    torch.testing.assert_close(out, ref)
