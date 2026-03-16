import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_maxpool_sum_scale_kernel(
    x_ptr,              # float32[B, IN_F]
    w_ptr,              # float32[OUT_F, IN_F]
    b_ptr,              # float32[OUT_F]
    out_ptr,            # float32[B]
    B: tl.constexpr,    # batch size
    IN_F,               # in_features (int)
    OUT_F,              # out_features (int)
    KERNEL,             # kernel_size (int), stride == kernel_size
    X_STRIDE,           # stride for x rows (int)
    W_ROW_STRIDE,       # weight row stride (int) -> typically IN_F
    W_COL_STRIDE,       # weight col stride (int) -> typically 1
    SCALE,              # scale factor (float)
    BLOCK_IN: tl.constexpr,   # tile size along IN_F
    BLOCK_KO: tl.constexpr,   # tile size along kernel window outputs
):
    pid = tl.program_id(axis=0)
    # Each program handles one row (batch element)
    x_row_ptr = x_ptr + pid * X_STRIDE

    # Number of non-overlapping pooling windows
    windows_count = OUT_F // KERNEL

    sum_acc = 0.0
    win = 0
    # Iterate over windows
    while win < windows_count:
        base_o = win * KERNEL
        # Track maximum over the current window
        window_max = -float("inf")

        ko_off = 0
        # Process the K outputs of the window in chunks of BLOCK_KO
        while ko_off < KERNEL:
            offs_k = tl.arange(0, BLOCK_KO)
            o_idx = base_o + ko_off + offs_k
            mask_o = o_idx < (base_o + KERNEL)

            # Accumulators for dot products for this chunk [BLOCK_KO]
            acc = tl.zeros([BLOCK_KO], dtype=tl.float32)

            m_off = 0
            # Reduce over input features dimension
            while m_off < IN_F:
                offs_m = tl.arange(0, BLOCK_IN)
                m_idx = m_off + offs_m
                mask_m = m_idx < IN_F

                # Load x chunk [BLOCK_IN]
                x_vec = tl.load(x_row_ptr + m_idx, mask=mask_m, other=0.0)

                # Load weight sub-matrix [BLOCK_KO, BLOCK_IN]
                w_ptrs = w_ptr + (o_idx[:, None] * W_ROW_STRIDE) + (m_idx[None, :] * W_COL_STRIDE)
                w_block = tl.load(w_ptrs, mask=mask_o[:, None] & mask_m[None, :], other=0.0)

                # FMA reduction over input features for each output in the window chunk
                acc += tl.sum(w_block * x_vec[None, :], axis=1)

                m_off += BLOCK_IN

            # Add bias
            b_vec = tl.load(b_ptr + o_idx, mask=mask_o, other=0.0)
            val_vec = acc + b_vec

            # Compute max over valid elements in this chunk and update window max
            chunk_max = tl.max(tl.where(mask_o, val_vec, -float("inf")), axis=0)
            window_max = tl.maximum(window_max, chunk_max)

            ko_off += BLOCK_KO

        # Accumulate sum over window maxima
        sum_acc += window_max
        win += 1

    # Scale and store result
    out_val = sum_acc * SCALE
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)  # kept for state/compat; computation is fused
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Fallback to PyTorch path for non-CUDA tensors
        if not x.is_cuda:
            x = self.matmul(x)
            x = self.max_pool(x.unsqueeze(1)).squeeze(1)
            x = torch.sum(x, dim=1)
            x = x * self.scale_factor
            return x

        # Ensure contiguous tensors
        x = x.contiguous()
        W = self.matmul.weight.contiguous()
        b = self.matmul.bias
        if b is None:
            b = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
        else:
            b = b.contiguous()

        B = x.shape[0]
        IN_F = x.shape[1]
        OUT_F = W.shape[0]
        KERNEL = self.max_pool.kernel_size if isinstance(self.max_pool.kernel_size, int) else self.max_pool.kernel_size[0]

        # Output tensor
        out = torch.empty(B, device=x.device, dtype=torch.float32)

        # Launch kernel: one program per batch row
        grid = (B,)

        # Choose tile sizes
        BLOCK_IN = 128
        BLOCK_KO = 32

        _linear_maxpool_sum_scale_kernel[grid](
            x, W, b, out,
            B,
            IN_F,
            OUT_F,
            KERNEL,
            x.stride(0),
            W.stride(0),
            W.stride(1),
            float(self.scale_factor),
            BLOCK_IN=BLOCK_IN,
            BLOCK_KO=BLOCK_KO,
            num_warps=4,
            num_stages=2,
        )
        return out

batch_size = 128
in_features = 10
out_features = 5
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]