"""
Ascend Triton sinkhorn_normalize (doubly-stochastic iteration).

Reference: 4_ori.py::Model
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.runtime.driver as driver

_NUM_VECTORCORES: int | None = None


def _num_vectorcores() -> int:
    global _NUM_VECTORCORES
    if _NUM_VECTORCORES is None:
        device = torch.npu.current_device()
        _NUM_VECTORCORES = int(
            driver.active.utils.get_device_properties(device)["num_vectorcore"]
        )
    return _NUM_VECTORCORES


def _is_accel_device(device: torch.device) -> bool:
    return device.type in ("npu", "cuda")


@triton.jit
def _sinkhorn_2d_kernel(
    x_ptr,
    y_ptr,
    stride_b0,
    stride_b1,
    stride_m,
    stride_n,
    B0: tl.constexpr,
    B1: tl.constexpr,
    MH: tl.constexpr,
    REP: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tasks = B0 * B1
    rows = tl.arange(0, MH)
    cols = tl.arange(0, MH)

    # Map tasks to vector cores using 1D stride loop: pid → range(pid, n_tasks, NUM_CORES)
    # This ensures balanced load and matches physical core topology while avoiding UB overflow
    for task in range(pid, n_tasks, NUM_CORES):
        b0 = task // B1
        b1 = task % B1
        base_off = b0 * stride_b0 + b1 * stride_b1
        ptrs = x_ptr + base_off + rows[:, None] * stride_m + cols[None, :] * stride_n
        x = tl.load(ptrs)

        # Use fp32 accumulation path for all reductions
        x_f32 = x.to(tl.float32)
        row_max = tl.max(x_f32, axis=1)
        x_f32 = x_f32 - row_max[:, None]
        ex = tl.exp(x_f32)
        row_sum = tl.sum(ex, axis=1)
        y_f32 = ex / row_sum[:, None]
        y_f32 = y_f32 + EPS

        for i in tl.static_range(0, REP):
            if i > 0:
                row_sum2 = tl.sum(y_f32, axis=1)
                y_f32 = y_f32 / (row_sum2[:, None] + EPS)
            col_sum2 = tl.sum(y_f32, axis=0)
            y_f32 = y_f32 / (col_sum2[None, :] + EPS)

        # Cast back to input dtype only at store
        y_out = y_f32.to(x_ptr.dtype.element_ty)
        out_ptrs = y_ptr + base_off + rows[:, None] * stride_m + cols[None, :] * stride_n
        tl.store(out_ptrs, y_out)


def _sinkhorn_torch(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    y = torch.softmax(x, dim=-1)
    y = y + eps
    y = y / (y.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        y = y / (y.sum(-1, keepdim=True) + eps)
        y = y / (y.sum(-2, keepdim=True) + eps)
    return y


def _can_use_triton_kernel(x: torch.Tensor, repeat: int) -> bool:
    return (
        x.ndim == 4
        and x.dtype == torch.float32
        and x.shape[-1] == x.shape[-2]
        and repeat >= 1
    )


def sinkhorn_normalize_triton(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
    *,
    force_triton: bool = True,
) -> torch.Tensor:
    """
    Sinkhorn normalize on NPU/CUDA.

    Default (force_triton=True): always launch Triton on npu/cuda when shape is supported.
    Set force_triton=False or env OP4_FORCE_TRITON=0 for PyTorch reference path.
    """
    env_force = os.environ.get("OP4_FORCE_TRITON", "1").lower()
    use_triton = force_triton and env_force not in ("0", "false", "no")

    if not use_triton:
        return _sinkhorn_torch(x, repeat, eps)

    if not _can_use_triton_kernel(x, repeat):
        raise ValueError(
            f"sinkhorn_normalize_triton expects 4D float32 square last dims, repeat>=1; "
            f"got shape={tuple(x.shape)} dtype={x.dtype} repeat={repeat}"
        )

    device = x.device
    if not _is_accel_device(device):
        raise RuntimeError(
            f"sinkhorn_normalize_triton requires npu/cuda when force_triton=True, got {device}"
        )

    if not x.is_contiguous():
        x = x.contiguous()

    b0, b1, mhc_m, mhc_n = x.shape
    y = torch.empty_like(x)
    s0, s1, s2, s3 = x.stride()

    if device.type == "npu":
        num_cores = _num_vectorcores()
        # Use 1D grid matching vector core count, with stride-loop dispatch inside kernel
        grid = (num_cores,)
    else:
        num_cores = 1
        grid = (b0 * b1,)

    _sinkhorn_2d_kernel[grid](
        x,
        y,
        s0,
        s1,
        s2,
        s3,
        B0=b0,
        B1=b1,
        MH=mhc_n,
        REP=repeat,
        EPS=eps,
        NUM_CORES=num_cores,
        num_warps=1,
        num_stages=1,
    )
    return y


class ModelTriton(nn.Module):
    """
    Triton-accelerated sinkhorn_normalize on Ascend NPU (force Triton by default).

    Iteratively normalizes a matrix to be doubly stochastic:
      1. softmax(x, dim=-1) + eps
      2. column-normalize: x / (x.sum(-2) + eps)
      3. repeat (row-normalize then column-normalize) for repeat-1 iterations
    """

    def __init__(self, repeat: int = 10, eps: float = 1e-6, *, force_triton: bool = True):
        super().__init__()
        self.repeat = repeat
        self.eps = eps
        self.force_triton = force_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sinkhorn_normalize_triton(
            x, self.repeat, self.eps, force_triton=self.force_triton
        )


ModelNew = ModelTriton

n0 = 1
n1 = 1024
mhc = 4


def get_inputs():
    x = torch.randn(n0, n1, mhc, mhc)
    return [x]


def get_init_inputs():
    return []
