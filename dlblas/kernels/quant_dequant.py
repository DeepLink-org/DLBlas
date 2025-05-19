# Copyright (c) 2025, DeepLink.
import torch
import triton
import triton.language as tl


def per_channel_quant_fp8(x, quant_dtype=torch.float8_e4m3fn):
    from torchao.float8.float8_utils import to_fp8_saturated
    x_amax = x.abs().amax(-1, True)
    x_scales = x_amax.float() / torch.finfo(quant_dtype).max
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    return x_quanted, x_scales


def per_channel_dequant_bf16(x_quanted, x_scales, dequant_dtype=torch.bfloat16):
    x_origin = x_quanted.float() * x_scales
    return x_origin.to(dequant_dtype)


# copy from
# https://github.com/deepseek-ai/DeepGEMM/blob/bd2a77552886b98c205af12f8d7d2d61247c4b27/deep_gemm/jit_kernels/utils.py#L58
def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    from deep_gemm import ceil_div
    return ceil_div(x, alignment) * alignment


@triton.jit
def _tma_align_input_scale_kernel(
    input_scale_ptr,
    output_ptr,
    m,
    k_div_block_size,
    input_scale_stride_m,
    input_scale_stride_k,
    output_stride_m,
    output_stride_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    grid_m = tl.num_programs(0)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    for m_base in range(pid_m, m, grid_m):
        input_offset = (input_scale_ptr + m_base * input_scale_stride_m + k_offsets * input_scale_stride_k)
        input_data = tl.load(input_offset, mask=k_offsets < k_div_block_size)
        output_offset = (output_ptr + k_offsets * output_stride_k + m_base * output_stride_m)
        tl.store(output_offset, input_data, mask=k_offsets < k_div_block_size)


# copy from lightllm
def tma_align_input_scale(input_scale: torch.Tensor):
    assert input_scale.dim() == 2
    m, k_div_block_size = input_scale.shape
    padd_m = get_tma_aligned_size(m, input_scale.element_size())
    output = torch.empty((k_div_block_size, padd_m), dtype=input_scale.dtype, device=input_scale.device)
    grid_m = min(m, 8192)
    BLOCK_SIZE_K = triton.next_power_of_2(k_div_block_size)
    _tma_align_input_scale_kernel[(grid_m, )](
        input_scale_ptr=input_scale,
        output_ptr=output,
        m=m,
        k_div_block_size=k_div_block_size,
        input_scale_stride_m=input_scale.stride(0),
        input_scale_stride_k=input_scale.stride(1),
        output_stride_m=output.stride(1),  # Note: these are swapped
        output_stride_k=output.stride(0),  # for column-major
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return output.t()[:m]
