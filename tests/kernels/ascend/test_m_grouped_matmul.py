import pytest
import torch
import random
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.m_grouped_gemm import m_grouped_gemm

device_ = infer_device()

def torch_grouped_matmul(a, b, size_per_group, trans_b):
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

@pytest.mark.parametrize(['N', 'K'],
[(4096, 4096), (512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('groups', [128])
@pytest.mark.parametrize('trans_b', [True, False])
def test_m_grouped_gemm(groups, N, K, dtype, trans_b):
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*1280)).to(device_).to(torch.int64)
    M = batch_sizes.sum().item()
    a = torch.randn(M, K, dtype = dtype, device = device_).view(-1, K)
    b = torch.randn(groups, N, K, dtype = dtype, device = device_) if trans_b else torch.randn(groups, K, N, dtype = dtype, device = device_)
    golden = torch_grouped_matmul(a, b, batch_sizes, trans_b)
    result = m_grouped_gemm(a, b, batch_sizes, trans_b)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)