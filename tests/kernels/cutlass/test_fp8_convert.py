# Copyright (c) 2025, DeepLink.
import ctypes
import subprocess
from typing import Any, Dict

import torch

ctype_map: Dict[Any, Any] = {
    **{
        t: getattr(ctypes, f'c_{t.__name__}')
        for t in (bool, int, float)
    },
    **{
        t: ctypes.c_void_p
        for t in (torch.int, torch.float, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.cuda.Stream)
    },
}


def map_ctype(value: Any) -> Any:
    ctype = ctype_map[value.dtype if isinstance(value, torch.Tensor) else type(value)]
    if isinstance(value, torch.Tensor):
        return ctype(value.data_ptr())
    if isinstance(value, torch.cuda.Stream):
        return ctype(value.cuda_stream)
    return ctype(value)


class TestFp8Convert:

    def test_fp8_convert(self):
        subprocess.check_call(
            'nvcc ../../../dlblas/kernels/cutlass/fp8_convert.cu -I../../../dlblas/third_party/cutlass/include -shared -std=c++17 --compiler-options=-fPIC -o ../../../dlblas/kernels/cutlass/fp8_convert.so',
            shell=True)
        lib = ctypes.CDLL('../../../dlblas/kernels/cutlass/fp8_convert.so')
        inp = torch.rand(1).to(torch.float8_e5m2)
        out = torch.zeros(1, device='cuda')
        out = out.to(torch.float8_e4m3fn)
        lib.launch(map_ctype(inp.cuda()), map_ctype(out))
        assert torch.allclose(inp.to(torch.float), out.to(torch.float).cpu(), atol=1, rtol=0.01)
