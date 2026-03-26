import torch
import triton
import triton.language as tl
from torch import Tensor
import os, importlib.util


# Robust import shim for utils.get_device_props
def _get_device_props():
    try:
        from utils import get_device_props  # type: ignore
        return get_device_props
    except Exception:
        base_path = os.environ.get("KERNELBENCH_ROOT", None)
        if base_path is None:
            raise ValueError("KERNELBENCH_ROOT is not set")
        utils_path = os.path.join(base_path, "lmdeploy/utils.py")
        spec = importlib.util.spec_from_file_location("lmdeploy_utils", utils_path)
        _utils = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(_utils)
        return _utils.get_device_props  # type: ignore


get_device_props = _get_device_props()


@triton.jit
def _compute_rms_norm(x, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf * tl.math.rsqrt(var + eps)
    out = (w * out).to(x.dtype)
    return out


@triton.jit
def rms_norm_kernel(input, weight, output, seq_len, input_row_stride: tl.constexpr, eps: tl.constexpr,
                    N_COLS: tl.constexpr, BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr):
    prog_id = tl.program_id(0)
    prog_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight + offsets, mask=mask).to(tl.float32)

    x_ptr = input + prog_id * input_row_stride + offsets
    out_ptr = output + prog_id * input_row_stride + offsets
    for _ in tl.range(prog_id, seq_len, prog_stride, num_stages=NUM_STAGES):
        x = tl.load(x_ptr, mask=mask)
        out = _compute_rms_norm(x, w, eps, N_COLS)
        tl.store(out_ptr, out, mask=mask)
        x_ptr += prog_stride * input_row_stride
        out_ptr += prog_stride * input_row_stride


@triton.jit
def add_rms_norm_kernel(input, weight, residual, output, out_residual, seq_len, input_row_stride: tl.constexpr,
                        residual_row_stride: tl.constexpr, eps: tl.constexpr, N_COLS: tl.constexpr,
                        BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr):
    prog_id = tl.program_id(0)
    prog_stride = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight + offsets, mask=mask).to(tl.float32)

    x_ptr = input + prog_id * input_row_stride + offsets
    res_ptr = residual + prog_id * residual_row_stride + offsets
    out_res_ptr = out_residual + prog_id * residual_row_stride + offsets
    out_ptr = output + prog_id * input_row_stride + offsets
    for _ in tl.range(prog_id, seq_len, prog_stride, num_stages=NUM_STAGES):
        x = tl.load(x_ptr, mask=mask)
        res = tl.load(res_ptr, mask=mask)
        new_x = x + res
        tl.store(out_res_ptr, new_x, mask=mask)
        out = _compute_rms_norm(new_x, w, eps, N_COLS)
        tl.store(out_ptr, out, mask=mask)
        x_ptr += prog_stride * input_row_stride
        res_ptr += prog_stride * input_row_stride
        out_ptr += prog_stride * input_row_stride
        out_res_ptr += prog_stride * input_row_stride


def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             eps: float = 1e-6,
             residual: Tensor = None,
             out: Tensor = None,
             out_residual: Tensor = None):
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    feat_size = weight.shape[0]
    assert hidden_states.size(-1) == feat_size
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)

    props = get_device_props(hidden_states.device.index)
    num_sm = props['multi_processor_count']
    warps_per_sm = props['warps_per_sm']
    blocks_per_sm = props['blocks_per_sm']
    num_warps = min(triton.cdiv(BLOCK_N, 128), 4)
    cta_per_sm = min(blocks_per_sm, warps_per_sm // num_warps)
    cta_per_device = num_sm * cta_per_sm
    num_stages = min(5, triton.cdiv(seq_len, cta_per_device))

    if out is None:
        out = torch.empty_like(hidden_states)

    grid = (min(seq_len, cta_per_device), )
    if residual is None:
        rms_norm_kernel[grid](
            hidden_states,
            weight,
            out,
            seq_len=seq_len,
            input_row_stride=input_stride,
            eps=eps,
            N_COLS=feat_size,
            BLOCK_N=BLOCK_N,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out
    else:
        if out_residual is None:
            out_residual = torch.empty_like(hidden_states)

        res_stride = residual.stride(-2)
        add_rms_norm_kernel[grid](
            hidden_states,
            weight,
            residual,
            out,
            out_residual,
            seq_len=seq_len,
            input_row_stride=input_stride,
            residual_row_stride=res_stride,
            eps=eps,
            N_COLS=feat_size,
            BLOCK_N=BLOCK_N,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out, out_residual


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        return rms_norm(hidden_states, weight, eps=self.eps, residual=residual)

