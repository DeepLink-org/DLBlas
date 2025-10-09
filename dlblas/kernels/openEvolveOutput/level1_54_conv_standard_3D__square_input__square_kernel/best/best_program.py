import triton
import triton.language as tl

@triton.jit
def _conv2d(
    x_ptr, w_ptr, y_ptr,
    stride_h, stride_w, padding_h, padding_w,
    n, c, h, w, k, r, s,  # input and kernel dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    out_h = (h + 2 * padding_h - r) // stride_h + 1
    out_w = (w + 2 * padding_w - s) // stride_w + 1
    total_output = n * k * out_h * out_w
    
    if pid >= total_output:
        return
    
    # Decompose PID into output indices
    output_channel_size = k * out_h * out_w
    n_index = pid // output_channel_size
    k_index = (pid % output_channel_size) // (out_h * out_w)
    oh_index = (pid % (out_h * out_w)) // out_w
    ow_index = pid % out_w

    accumulator = 0.0
    # Iterate over input channels and kernel positions
    for c_index in range(c):
        for rr in range(r):
            for ss in range(s):
                # Calculate input position with padding
                h_in = oh_index * stride_h - padding_h + rr
                w_in = ow_index * stride_w - padding_w + ss
                
                # Check input boundaries
                if h_in >= 0 and h_in < h and w_in >= 0 and w_in < w:
                    x_offset = n_index * c * h * w + c_index * h * w + h_in * w + w_in
                    x_val = tl.load(x_ptr + x_offset)
                else:
                    x_val = 0.0
                
                w_offset = k_index * c * r * s + c_index * r * s + rr * s + ss
                w_val = tl.load(w_ptr + w_offset)
                
                accumulator += x_val * w_val
    
    tl.store(y_ptr + pid, accumulator)