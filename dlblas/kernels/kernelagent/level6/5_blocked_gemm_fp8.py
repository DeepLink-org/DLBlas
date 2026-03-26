import torch
import triton
import triton.language as tl
from packaging import version
import os, importlib.util
from torch import Tensor
import functools

TRITON_VERSION = version.parse(triton.__version__)

if TRITON_VERSION >= version.parse('3.0.0'):
    fast_expf = tl.math.exp
else:
    fast_expf = tl.math.fast_expf


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == 'cuda'


@functools.lru_cache
def supports_tma():
    ret = is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    if not ret:
        return False

    VALID_VERSION = version.parse('3.4.0')
    return TRITON_VERSION == VALID_VERSION


if supports_tma():
    from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401



# Robust import shim for utils.get_device_props (works by package or file path)
try:
    from utils import get_device_props  # type: ignore
except Exception:
    base_path = os.environ.get("KERNELBENCH_ROOT", None)
    if base_path is None:
        raise ValueError("KERNELBENCH_ROOT is not set")
    utils_path = os.path.join(base_path, "lmdeploy/utils.py")
    spec = importlib.util.spec_from_file_location("lmdeploy_utils", utils_path)
    _utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(_utils)
    get_device_props = _utils.get_device_props  # type: ignore


def _gemm_fp8_tma_pre_hook(nargs):
    BLOCK_M = nargs['BLOCK_M']
    BLOCK_N = nargs['BLOCK_N']
    BLOCK_K = nargs['BLOCK_K']
    nargs['desc_a'].block_shape = (BLOCK_M, BLOCK_K)
    nargs['desc_b'].block_shape = (BLOCK_N, BLOCK_K)


@triton.autotune(configs=[
    triton.Config({
        'BLOCK_M': 128,
        'BLOCK_N': 128,
    }, num_stages=3, num_warps=8, pre_hook=_gemm_fp8_tma_pre_hook),
    triton.Config({
        'BLOCK_M': 128,
        'BLOCK_N': 64,
    }, num_stages=3, num_warps=4, pre_hook=_gemm_fp8_tma_pre_hook)
],
                 key=['N', 'K'])
@triton.jit
def _gemm_fp8_tma_kernel(
    desc_a,
    a_scale_ptr,
    desc_b,
    b_scale_ptr,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Gemm fp8 kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M

    offs_bsn = pid_n * BLOCK_N // group_bn
    as_ptrs = a_scale_ptr + offs_am * stride_asm
    bs_ptrs = b_scale_ptr + offs_bsn * stride_bsn

    acc_scale = tl.load(as_ptrs) * tl.load(bs_ptrs)
    acc_ratio = 1 / acc_scale
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_k = 0
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # load scales
        k_start = (k + 1) * BLOCK_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=k_start < K, other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = desc_a.load([off_m, off_k])
        b = desc_b.load([off_n, off_k]).T

        # mma
        accumulator = tl.dot(a, b, acc=accumulator * acc_ratio[:, None])

        # update scales and ratio
        new_acc_scale = a_scale * b_scale
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        off_k += BLOCK_K
    c = accumulator * (acc_ratio * acc_scale)[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(configs=[
    triton.Config({
        'BLOCK_M': 64,
        'BLOCK_N': 128,
    }, num_stages=3, num_warps=4),
    triton.Config({
        'BLOCK_M': 128,
        'BLOCK_N': 64,
    }, num_stages=3, num_warps=4)
],
                 key=['N', 'K'])
@triton.jit
def _gemm_fp8_kernel(
    A,
    a_scale_ptr,
    B,
    b_scale_ptr,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_am,
    stride_ak: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Gemm fp8 kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = pid_n * BLOCK_N // group_bn
    as_ptrs = a_scale_ptr + offs_am * stride_asm
    bs_ptrs = b_scale_ptr + offs_bsn * stride_bsn

    acc_scale = tl.load(as_ptrs) * tl.load(bs_ptrs)
    acc_ratio = 1 / acc_scale
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # load scales
        k_start = (k + 1) * BLOCK_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=k_start < K, other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # mma
        accumulator = tl.dot(a, b, acc=accumulator * acc_ratio[:, None])

        # update scales and ratio
        new_acc_scale = a_scale * b_scale
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator * (acc_ratio * acc_scale)[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def blocked_gemm_fp8(A: Tensor,
                     A_scale: Tensor,
                     B: Tensor,
                     B_scale: torch.Tensor,
                     out_dtype: torch.dtype = torch.float16):
    """Gemm fp8."""

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    assert A.dim() == 2
    assert A_scale.dim() == 2
    assert B.dim() == 2
    assert B_scale.dim() == 2

    M, K = A.shape
    _, N = B.shape

    group_ak = triton.cdiv(K, A_scale.size(1))
    group_bk = triton.cdiv(K, B_scale.size(0))
    group_bn = triton.cdiv(N, B_scale.size(1))

    C = A.new_empty(M, N, dtype=out_dtype)

    BLOCK_K = max(group_ak, group_bk)

    run_tma = supports_tma()
    run_tma = run_tma and A.is_contiguous() and B.T.is_contiguous()

    # run_tma = False
    if run_tma:

        dummy_block = (1, 1)
        desc_a = TensorDescriptor.from_tensor(A, block_shape=dummy_block)
        desc_b = TensorDescriptor.from_tensor(B.T, block_shape=dummy_block)

        def _grid_tma(META):
            """Grid tma."""
            BLOCK_M = META['BLOCK_M']
            BLOCK_N = META['BLOCK_N']
            return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

        _gemm_fp8_tma_kernel[_grid_tma](
            desc_a,
            A_scale,
            desc_b,
            B_scale,
            C,
            M=M,
            N=N,
            K=K,
            group_ak=group_ak,
            group_bk=group_bk,
            group_bn=group_bn,
            stride_asm=A_scale.stride(0),
            stride_ask=A_scale.stride(1),
            stride_bsk=B_scale.stride(0),
            stride_bsn=B_scale.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            BLOCK_K=BLOCK_K,
            GROUP_M=8,
        )
    else:
        _gemm_fp8_kernel[grid](
            A,
            A_scale,
            B,
            B_scale,
            C,
            M=M,
            N=N,
            K=K,
            group_ak=group_ak,
            group_bk=group_bk,
            group_bn=group_bn,
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_asm=A_scale.stride(0),
            stride_ask=A_scale.stride(1),
            stride_bk=B.stride(0),
            stride_bn=B.stride(1),
            stride_bsk=B_scale.stride(0),
            stride_bsn=B_scale.stride(1),
            stride_cm=C.stride(0),
            stride_cn=C.stride(1),
            BLOCK_K=BLOCK_K,
            GROUP_M=8,
        )

    return C


class ModelNew(torch.nn.Module):
    def __init__(self, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    def forward(self, A: Tensor, A_scale: Tensor, B: Tensor, B_scale: Tensor):
        return blocked_gemm_fp8(A, A_scale, B, B_scale, out_dtype=self.out_dtype)

