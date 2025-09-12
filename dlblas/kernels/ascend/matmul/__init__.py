import torch
from .matmul_v1 import register as triton_matmul_register_v1
from .matmul_v2 import register as triton_matmul_register_v2
from dlblas.op_registry import op_registry


op_name = 'matmul_ascend'
def call(a:torch.Tensor, b:torch.Tensor):
    m, k = a.shape
    k, n = b.shape
    cache_key = f'{op_name}_k={k}_n={n}'
    op = op_registry.get_op(op_name, (a, b), cache_key=cache_key)
    return op(a, b)

triton_matmul_register_v1(op_name)
triton_matmul_register_v2(op_name)
assert op_registry.get_op_count(name=op_name) == 2