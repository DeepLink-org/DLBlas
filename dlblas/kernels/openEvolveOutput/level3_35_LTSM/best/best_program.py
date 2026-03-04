import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, b_ptr, output_ptr,
    # Matrix dimensions
    in_features, out_features,
    # Strides for x
    stride_xb, stride_xf,
    # Strides for weight
    stride_wo, stride_wi,
    # Strides for output
    stride_outb, stride_outf,
    # Block size for features (must be power of two)
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # Batch index
    pid_o = tl.program_id(1)  # Output feature index

    # Create a range of indices for the input features (from 0 to BLOCK_SIZE-1)
    idx = tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent out-of-bounds accesses
    mask = idx < in_features

    # Calculate the pointer offsets for the input row (batch index = pid_b)
    x_offset = pid_b * stride_xb + idx * stride_xf
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

    # Calculate the pointer offsets for the weight column (output feature index = pid_o)
    w_offset = pid_o * stride_wo + idx * stride_wi
    w = tl.load(w_ptr + w_offset, mask=mask, other=0.0)

    # Compute the dot product
    dot = tl.sum(x * w)

    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_o)
        dot += bias

    # Write the result to the output
    output_offset = pid_b * stride_outb + pid_o * stride_outf
    tl.store(output_ptr + output_offset, dot)