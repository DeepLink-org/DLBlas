import importlib
import pytest
import torch
import random
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.grouped_gemm import m_grouped_gemm, k_grouped_gemm, grouped_gemm_triton

DEV = infer_device()

def m_grouped_matmul_torch(a, b, size_per_group, trans_b):
    b = torch.transpose(b, 1, 2).contiguous() if trans_b else b
    return torch.ops.npu.npu_grouped_matmul(
        [a],
        [b],
        bias=None,
        group_list=size_per_group.cumsum(0),
        split_item=2,
        group_type=0,
        group_list_type=0,
    )[0]


def k_grouped_matmul_torch(a, b, batch_sizes):
    K, M = a.shape
    K_, N = b.shape
    assert a.stride(-1) == 1, "Please make sure A is K-major"
    assert b.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    num_groups = batch_sizes.shape[0]
    out = a.new_empty(num_groups, M, N)
    group_end = batch_sizes.cumsum(0) - batch_sizes + batch_sizes
    group_start = batch_sizes.cumsum(0) - batch_sizes
    for g, (start, end) in enumerate(zip(group_start, group_end)):
        rhs = b[start:end, :]
        lhs = a[start:end, :]
        out[g] = lhs.T @ rhs
    return out


def grouped_gemm_npu(x: torch.Tensor, weights: torch.Tensor, split_sizes: torch.Tensor) -> torch.Tensor:
    from mindspeed.core.fusions.grouped_matmul import Ops
    weights = weights.transpose(1, 2)
    out = Ops.gmm(x, weights, split_sizes, trans_b=False)
    return out


def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    lst = [0] * length
    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, 2*int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return lst


@pytest.mark.parametrize(['N', 'K'], [(4096, 4096), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('groups', [8])
@pytest.mark.parametrize('trans_b', [False, True])
def test_m_grouped_gemm(groups, N, K, dtype, trans_b):
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*2560)).to(DEV).to(torch.int64)
    M = batch_sizes.sum().item()
    a = torch.randn(M, K, dtype = dtype, device = DEV)
    b = torch.randn(groups, N, K, dtype = dtype, device = DEV) if trans_b else torch.randn(groups, K, N, dtype = dtype, device = DEV)
    golden = m_grouped_matmul_torch(a, b, batch_sizes, trans_b)
    result = m_grouped_gemm(a, b, batch_sizes, trans_b)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)


@pytest.mark.parametrize(['M', 'N'], [(512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('groups', [8])
def test_k_grouped_gemm(groups, M, N, dtype):
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*2560)).to(DEV).to(torch.int64).abs()
    K = batch_sizes.sum().item()
    a = torch.randn(K, M, dtype = dtype, device = DEV)
    b = torch.randn(K, N, dtype = dtype, device = DEV)
    golden = k_grouped_matmul_torch(a, b, batch_sizes.cpu())
    result = k_grouped_gemm(a, b, batch_sizes)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)


@pytest.mark.skipif(not importlib.util.find_spec("mindspeed"), reason="requires mindspeed")
@pytest.mark.parametrize(['N', 'K'], [(4096, 4096), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('groups', [8])
def test_grouped_gemm(groups, N, K, dtype):
    split_sizes_triton = torch.Tensor(generate_random_list(groups, groups*2560)).to(DEV).to(torch.int64)
    M = split_sizes_triton.sum().item()
    x_triton = torch.randn(M, K, dtype = dtype, device = DEV, requires_grad=True)
    w_triton = torch.randn(groups, N, K, dtype = dtype, device = DEV, requires_grad=True)
    with torch.no_grad():
        x_torch = x_triton.clone()
        w_torch = w_triton.clone()
        split_sizes_torch = split_sizes_triton.clone()
    x_torch.requires_grad = True
    w_torch.requires_grad = True
    out_triton = grouped_gemm_triton(x_triton, w_triton, split_sizes_triton)
    out_torch = grouped_gemm_npu(x_torch, w_torch, split_sizes_torch)
    mask = out_torch.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    torch.testing.assert_close(out_triton[mask], out_torch[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(out_triton[~mask], out_torch[~mask], atol = 0, rtol = tmprtol)

    loss_triton = torch.sum(torch.mean(out_triton))
    loss_torch = torch.sum(torch.mean(out_torch))
    assert torch.allclose(loss_torch, loss_triton)
    # for backward
    dout_torch = torch.randn_like(loss_torch)
    with torch.no_grad():
        dout_triton = dout_torch.clone()
    loss_torch.backward(dout_torch, retain_graph=True)
    loss_triton.backward(dout_triton, retain_graph=True)
    assert torch.allclose(x_torch.grad, x_triton.grad, rtol=1e-8, atol=1e-8)
    assert torch.allclose(w_torch.grad, w_triton.grad, rtol=1e-8, atol=1e-8)
    