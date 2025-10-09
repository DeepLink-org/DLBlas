import torch
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    # Pointers to tensors
    x_ptr, w_ptr, bias_ptr, output_ptr,
    # Tensor dimensions
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    # Strides and paddings
    stride_h, stride_w,
    padding_h, padding_w,
    kernel_h, kernel_w,
    groups,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # We use a 3D grid: (batch, output channel, spatial_block)
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)  # which block of the spatial dimension

    # Calculate the spatial block start and offsets
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = offsets < (H_out * W_out)

    # Decompose the spatial offset into height and width indices
    offs_h = offsets // W_out
    offs_w = offsets % W_out

    # Group processing
    out_channels_per_group = C_out // groups
    in_channels_per_group = C_in // groups
    group_id = pid_oc // out_channels_per_group
    oc_in_group = pid_oc % out_channels_per_group
    ic_start = group_id * in_channels_per_group

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over kernel and input channels
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for ic in range(in_channels_per_group):
                # Calculate the input element indices
                input_h = (offs_h * stride_h) + kh - padding_h
                input_w = (offs_w * stride_w) + kw - padding_w

                # Check if the input indices are within bounds
                in_bounds = (input_h >= 0) & (input_h < H_in) & (input_w >= 0) & (input_w < W_in) & spatial_mask
                input_idx = ic_start + ic

                # Compute the input pointer offsets
                input_offsets = pid_b * (C_in * H_in * W_in) + input_idx * (H_in * W_in) + input_h * W_in + input_w

                # Load input values with masking
                input_val = tl.load(x_ptr + input_offsets, mask=in_bounds, other=0.0)

                # Weight offset: [input_channel, output_channel_in_group, kernel_h, kernel_w]
                weight_offset = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) + \
                               oc_in_group * (kernel_h * kernel_w) + \
                               kh * kernel_w + kw

                weight_val = tl.load(w_ptr + weight_offset)

                # Accumulate
                acc += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_oc)
        acc += bias_val

    # Compute output offsets: [batch, output_channel, height, width]
    output_offsets = pid_b * (C_out * H_out * W_out) + pid_oc * (H_out * W_out) + offsets
    tl.store(output_ptr + output_offsets, acc, mask=spatial_mask)