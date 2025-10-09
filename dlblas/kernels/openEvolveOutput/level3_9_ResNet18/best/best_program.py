import triton
import triton.language as tl

@triton.jit
def conv2d_7x7_naive(
    x_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, in_h, in_w,
    out_channels, out_h, out_w,
    stride, padding,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_OUT_CH: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # We'll use blocks for the output dimensions
    # Check boundaries
    if pid_batch >= batch_size or pid_out_ch >= out_channels or pid_h >= out_h or pid_w >= out_w:
        return

    # Offsets for the current output element
    out_offset = pid_batch * out_channels * out_h * out_w + pid_out_ch * out_h * out_w + pid_h * out_w + pid_w
    output = 0.0

    # Loop over in_channels and kernel (7x7)
    for c in range(in_channels):
        for kh in range(7):
            for kw in range(7):
                # input position
                h_in = pid_h * stride - padding + kh
                w_in = pid_w * stride - padding + kw
                if h_in >= 0 and h_in < in_h and w_in >= 0 and w_in < in_w:
                    input_offset = pid_batch * in_channels * in_h * in_w + c * in_h * in_w + h_in * in_w + w_in
                    weight_offset = pid_out_ch * in_channels * 7 * 7 + c * 7 * 7 + kh * 7 + kw
                    x_val = tl.load(x_ptr + input_offset)
                    w_val = tl.load(weight_ptr + weight_offset)
                    output += x_val * w_val

    tl.store(output_ptr + out_offset, output)