import pytest
import torch
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.matmul.matmul_v1 import call as matmul_v1
from dlblas.kernels.ascend.matmul.matmul_v2 import call as matmul_v2

device_ = infer_device()
@pytest.mark.parametrize(
['M', 'N', 'K'],
[
    (128, 128, 128),
    (4, 47, 31),  
    (4096, 4096, 4096),
    (512, 512, 512),
    (2048, 7168, 16384),
],)
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('triton_kernel', [matmul_v1, matmul_v2])
def test_matmul_v2(M, N, K, dtype, triton_kernel):
    mat_a = torch.randn([M, K], dtype = dtype, device = device_)
    mat_b = torch.randn([K, N], dtype = dtype, device = device_)
    result = triton_kernel(mat_a, mat_b)
    golden = torch.matmul(mat_a, mat_b)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)