import triton
import triton.language as tl

@triton.jit
def _logsumexp_kernel(
    x_ptr,
    output_ptr,
    x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    n_channels, n_height, n_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Total number of spatial locations and batches: n_batch * n_height * n_width
    pid = tl.program_id(0)
    total_spatial = n_height * n_width
    total_elements = total_spatial * tl.num_programs(0)  # Actually, we don't have the batch in the grid? We set the grid to [n_batch * n_height * n_width]. So we need to know n_batch? Actually, we don't pass n_batch. How do we get it? We can compute from the total number of elements.

    # Actually, we pass n_batch? We don't. We can compute the batch index by dividing by total_spatial.
    # But note: we set the grid to (n_batch * n_height * n_width,). So we can recover:
    batch_id = pid // total_spatial
    spatial_id = pid % total_spatial
    h_id = spatial_id // n_width
    w_id = spatial_id % n_width

    # Now, we want to load the entire channel for (batch_id, h_id, w_id)
    # We have a block of BLOCK_SIZE threads. We set BLOCK_SIZE to the next power of two >= n_channels? But we know n_channels, so we set it to 16? Actually, we can set BLOCK_SIZE as a constant and use a mask.

    # Create offsets for the channel dimension
    c_offsets = tl.arange(0, BLOCK_SIZE)
    mask = c_offsets < n_channels

    # Compute the base pointer for the batch and spatial location
    x_batch_ptr = x_ptr + batch_id * x_batch_stride + h_id * x_height_stride + w_id * x_width_stride
    # Now, for each channel, the pointer is x_batch_ptr + c_offsets * x_channel_stride
    x_ptrs = x_batch_ptr + c_offsets * x_channel_stride

    # Load the values
    x_vals = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # Find the max value
    max_val = tl.max(x_vals, axis=0)

    # Subtract the max and exponentiate
    exp_vals = tl.exp(x_vals - max_val)
    # Sum the exponentials
    sum_exp = tl.sum(exp_vals, axis=0)
    # Compute log and add max
    log_sum_exp = tl.log(sum_exp) + max_val

    # Now, store the result in the output tensor at [batch_id, 0, h_id, w_id]
    output_batch_ptr = output_ptr + batch_id * output_batch_stride + 0 * output_channel_stride + h_id * output_height_stride + w_id * output_width_stride
    tl.store(output_batch_ptr, log_sum_exp)

def triton_logsumexp(x, dim=1, keepdim=False):
    # We assume x is 4D and we are reducing over the channel dimension (dim=1) and keeping the spatial dimensions.
    # We also assume we want keepdim=True -> output has shape [batch, 1, height, width]
    assert x.dim() == 4
    assert dim == 1
    assert keepdim is True

    n_batch, n_channels, n_height, n_width = x.shape
    # Make sure the input is contiguous
    x_cont = x.contiguous()
    # Output tensor
    output = torch.empty(n_batch, 1, n_height, n_width, device=x.device, dtype=x.dtype)

    # Total number of elements in the grid: n_batch * n_height * n_width
    total_elements = n_batch * n_height * n_width

    # We set the block size to the next power of two that is >= n_channels? But we can set it to 32 (the next power of two after 16) to avoid bank conflicts? Actually, we can set it to 32 and then mask the extra channels.
    BLOCK_SIZE = 32  # must be power of two for reduction? Not strictly, but helps.

    # We need to compute the strides
    x_strides = x_cont.stride()
    output_strides = output.stride()

    # Launch kernel
    grid = (total_elements,)
    _logsumexp_kernel[grid](
        x_cont,
        output,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        n_channels, n_height, n_width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output