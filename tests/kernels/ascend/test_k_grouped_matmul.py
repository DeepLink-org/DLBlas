import pytest
import torch
import random
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.k_grouped_gemm import k_grouped_gemm

DEVICE = infer_device()

# def torch_grouped_matmul(a, b, size_per_group, trans_b):
#     b = torch.transpose(b, 1, 2).contiguous() if trans_b else b
#     return torch.ops.npu.npu_grouped_matmul(
#         [a],
#         [b],
#         bias=None,
#         group_list=size_per_group.cumsum(0),
#         split_item=2,
#         group_type=0,
#         group_list_type=0,
#     )[0]
def torch_grouped_matmul_dw(a, b, batch_sizes):
    K, M = a.shape
    K_, N = b.shape

    assert a.stride(-1) == 1, "Please make sure A is K-major"
    assert b.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    num_groups = batch_sizes.shape[0]

    out = a.new_empty(num_groups, M, N)

    group_end = batch_sizes.cumsum(0) - batch_sizes + batch_sizes
    group_start = batch_sizes.cumsum(0) - batch_sizes
    print(f"zcx: batch_size={batch_sizes}")
    print(f"zcx: group_start={group_start}")
    print(f"zcx: group_end={group_end}")
    for g, (start, end) in enumerate(zip(group_start, group_end)):
        rhs = b[start:end, :]
        lhs = a[start:end, :]
        out[g] = lhs.T @ rhs
    return out #.contiguous()


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


@pytest.mark.parametrize(['M', 'N'],
[(512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('groups', [128])
def test_k_grouped_gemm(groups, M, N, dtype):
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*1280)).to(DEVICE).to(torch.int64).abs()
    K = batch_sizes.sum().item()
    a = torch.randn(K, M, dtype = dtype, device = DEVICE)
    b = torch.randn(K, N, dtype = dtype, device = DEVICE)
    golden = torch_grouped_matmul_dw(a, b, batch_sizes.cpu())
    result = k_grouped_gemm(a, b, batch_sizes)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)