@triton.jit
def conv_transpose1d_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    in_channels, out_channels, kernel_size: tl.constexpr, stride, padding, dilation, length, output_length,
    stride_xb, stride_xc, stride_xl,
    stride_wic, stride_woc, stride_wk,
    stride_ob, stride_oc, stride_ol,
    BLOCK_SIZE: tl.constexpr
):
    pid_ol = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_oc = tl.program_id(2)
    
    ol_offsets = pid_ol * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ol_mask = ol_offsets < output_length
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for ic in range(in_channels):
        w_offsets = ic * stride_wic + pid_oc * stride_woc + tl.arange(0, kernel_size)
        weights = tl.load(weight_ptr + w_offsets)
        
        for k in range(kernel_size):
            input_pos = (ol_offsets + padding - k * dilation) // stride
            valid_mask = (ol_offsets + padding - k * dilation) % stride == 0
            in_bounds = (input_pos >= 0) & (input_pos < length)
            mask = valid_mask & in_bounds & ol_mask
            
            x_vals = tl.load(
                x_ptr + pid_b * stride_xb + ic * stride_xc + input_pos * stride_xl,
                mask=mask,
                other=0.0
            )
            acc += x_vals * weights[k]
    
    output_offsets = pid_b * stride_ob + pid_oc * stride_oc + ol_offsets * stride_ol
    tl.store(output_ptr + output_offsets, acc, mask=ol_mask)