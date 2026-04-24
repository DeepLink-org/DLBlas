import math

import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - exercised only when torch_npu is unavailable.
    torch_npu = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only when Triton is unavailable.
    triton = None
    tl = None


_TRITON_BLOCK_H = 128


def _get_default_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_block_h(hidden: int) -> int:
    block_h = math.gcd(_TRITON_BLOCK_H, hidden)
    if block_h == 0:
        raise ValueError(f"invalid hidden size: {hidden}")
    return block_h


if triton is not None:
    @triton.jit
    def _mhc_pre_apply_mix_kernel(
        x_ptr,
        mix_ptr,
        out_ptr,
        stride_xn,
        stride_xm,
        stride_xh,
        stride_mn,
        stride_mm,
        stride_on,
        stride_oh,
        BLOCK_H: tl.constexpr,
        NUM_H_TILES: tl.constexpr,
        MHC: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        for i_h_tile in range(NUM_H_TILES):
            offs_h = i_h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for i_mhc in range(MHC):
                x = tl.load(
                    x_ptr + pid_n * stride_xn + i_mhc * stride_xm + offs_h * stride_xh
                ).to(tl.float32)
                mix = tl.load(mix_ptr + pid_n * stride_mn + i_mhc * stride_mm)
                acc += x * mix
            tl.store(
                out_ptr + pid_n * stride_on + offs_h * stride_oh,
                acc.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            )


    @triton.jit
    def _mhc_post_kernel(
        layer_output_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        out_ptr,
        stride_lon,
        stride_loh,
        stride_rn,
        stride_rm,
        stride_rh,
        stride_pn,
        stride_pm,
        stride_cn,
        stride_ci,
        stride_co,
        stride_on,
        stride_om,
        stride_oh,
        BLOCK_H: tl.constexpr,
        NUM_H_TILES: tl.constexpr,
        MHC: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_i = tl.arange(0, MHC)
        offs_o = tl.arange(0, MHC)

        for i_h_tile in range(NUM_H_TILES):
            offs_h = i_h_tile * BLOCK_H + tl.arange(0, BLOCK_H)

            layer_output = tl.load(
                layer_output_ptr + pid_n * stride_lon + offs_h * stride_loh
            ).to(tl.float32)
            residual_block = tl.load(
                residual_ptr
                + pid_n * stride_rn
                + offs_i[:, None] * stride_rm
                + offs_h[None, :] * stride_rh
            ).to(tl.float32)
            comb_t = tl.load(
                comb_mix_ptr
                + pid_n * stride_cn
                + offs_o[:, None] * stride_co
                + offs_i[None, :] * stride_ci
            )
            term2 = tl.dot(comb_t, residual_block, input_precision="ieee", out_dtype=tl.float32)
            post_mix_block = tl.load(
                post_mix_ptr
                + pid_n * stride_pn
                + offs_o[:, None] * stride_pm
            )
            out_block = term2 + post_mix_block * layer_output[None, :]
            tl.store(
                out_ptr
                + pid_n * stride_on
                + offs_o[:, None] * stride_om
                + offs_h[None, :] * stride_oh,
                out_block.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            )


def _flatten_residual(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(-1, x.shape[-2], x.shape[-1])


def _flatten_mix(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(-1, x.shape[-2])


def _flatten_layer_output(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(-1, x.shape[-1])


def _multilayer_recompute_torch(
    initial_residual: torch.Tensor,
    pre_mix_list,
    layer_output_list,
    post_mix_list,
    comb_mix_list,
):
    layer_input_list = []
    residual_list = []

    residual = initial_residual
    for i in range(len(pre_mix_list)):
        layer_input = (residual * pre_mix_list[i]).sum(dim=2).to(torch.bfloat16)
        layer_input_list.append(layer_input)

        if i < len(layer_output_list):
            residual_fp32 = residual.float()
            comb = comb_mix_list[i].transpose(-1, -2).float()
            term2 = torch.matmul(comb, residual_fp32)
            residual = (
                layer_output_list[i].float().unsqueeze(2) * post_mix_list[i]
                + term2
            ).to(torch.bfloat16)
            residual_list.append(residual)

    return layer_input_list, residual_list


def _launch_pre_apply_mix(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    assert triton is not None and tl is not None
    x_3d = _flatten_residual(x)
    mix_2d = _flatten_mix(mix)
    n_tokens, mhc_mult, hidden = x_3d.shape
    block_h = _resolve_block_h(hidden)
    out_2d = torch.empty((n_tokens, hidden), device=x.device, dtype=torch.bfloat16)
    grid = (n_tokens,)
    _mhc_pre_apply_mix_kernel[grid](
        x_3d,
        mix_2d,
        out_2d,
        x_3d.stride(0),
        x_3d.stride(1),
        x_3d.stride(2),
        mix_2d.stride(0),
        mix_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        BLOCK_H=block_h,
        NUM_H_TILES=hidden // block_h,
        MHC=mhc_mult,
    )
    return out_2d.view(*x.shape[:-2], hidden)


def _launch_post(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    assert triton is not None and tl is not None
    layer_output_2d = _flatten_layer_output(layer_output)
    residual_3d = _flatten_residual(residual)
    post_mix_2d = _flatten_mix(post_mix)
    comb_mix_3d = comb_mix.contiguous().view(-1, comb_mix.shape[-2], comb_mix.shape[-1])
    n_tokens, mhc_mult, hidden = residual_3d.shape
    block_h = _resolve_block_h(hidden)
    out_3d = torch.empty_like(residual_3d)
    grid = (n_tokens,)
    _mhc_post_kernel[grid](
        layer_output_2d,
        residual_3d,
        post_mix_2d,
        comb_mix_3d,
        out_3d,
        layer_output_2d.stride(0),
        layer_output_2d.stride(1),
        residual_3d.stride(0),
        residual_3d.stride(1),
        residual_3d.stride(2),
        post_mix_2d.stride(0),
        post_mix_2d.stride(1),
        comb_mix_3d.stride(0),
        comb_mix_3d.stride(1),
        comb_mix_3d.stride(2),
        out_3d.stride(0),
        out_3d.stride(1),
        out_3d.stride(2),
        BLOCK_H=block_h,
        NUM_H_TILES=hidden // block_h,
        MHC=mhc_mult,
    )
    return out_3d.view_as(residual)


class ModelNew(nn.Module):
    """
    算子：multilayer_recompute
    功能：使用 Triton 在设备端逐层计算 layer_input，并在需要时更新 residual。
    输入：
        initial_residual: [batch, seq, mhc_mult, hidden]
        pre_mix_list: list[[batch, seq, mhc_mult, 1]]
        layer_output_list: list[[batch, seq, hidden]]
        post_mix_list: list[[batch, seq, mhc_mult, 1]]
        comb_mix_list: list[[batch, seq, mhc_mult, mhc_mult]]
    输出：
        layer_input_list: list[[batch, seq, hidden]]
        residual_list: list[[batch, seq, mhc_mult, hidden]]
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        initial_residual: torch.Tensor,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
    ):
        device_type = initial_residual.device.type
        if triton is None or device_type not in {"npu", "cuda"} or initial_residual.dtype != torch.bfloat16:
            return _multilayer_recompute_torch(
                initial_residual,
                pre_mix_list,
                layer_output_list,
                post_mix_list,
                comb_mix_list,
            )

        layer_input_list = []
        residual_list = []
        residual = initial_residual.contiguous()
        for i in range(len(pre_mix_list)):
            layer_input = _launch_pre_apply_mix(residual, pre_mix_list[i])
            layer_input_list.append(layer_input)

            if i < len(layer_output_list):
                residual = _launch_post(
                    layer_output_list[i],
                    residual,
                    post_mix_list[i],
                    comb_mix_list[i],
                )
                residual_list.append(residual)

        return layer_input_list, residual_list


batch_size = 1
seq_len = 8192
mhc_mult = 4
hidden = 2560
num_layers = 2
num_post = 1


def get_init_inputs():
    return []


def get_inputs():
    device = _get_default_device()
    initial_residual = torch.randn(
        batch_size, seq_len, mhc_mult, hidden, device=device, dtype=torch.bfloat16
    )
    pre_mix_list = [
        torch.randn(batch_size, seq_len, mhc_mult, 1, device=device, dtype=torch.float32)
        for _ in range(num_layers)
    ]
    layer_output_list = [
        torch.randn(batch_size, seq_len, hidden, device=device, dtype=torch.bfloat16)
        for _ in range(num_post)
    ]
    post_mix_list = [
        torch.randn(batch_size, seq_len, mhc_mult, 1, device=device, dtype=torch.float32)
        for _ in range(num_post)
    ]
    comb_mix_list = [
        torch.randn(batch_size, seq_len, mhc_mult, mhc_mult, device=device, dtype=torch.float32)
        for _ in range(num_post)
    ]
    return [initial_residual, pre_mix_list, layer_output_list, post_mix_list, comb_mix_list]
