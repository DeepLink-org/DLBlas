import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["NHW", "C_OUT"],
)
@triton.jit
def conv1x1_relu_kernel(
    x_ptr,        # *f32 [N, C_IN, H, W]
    w_ptr,        # *f32 [C_OUT, C_IN, 1, 1] contiguous
    b_ptr,        # *f32 [C_OUT]
    y_ptr,        # *f32 [N, C_TOT, H, W] (destination, may be larger than C_OUT)
    NHW,          # int: N * H * W
    H,            # int
    W,            # int
    Y_OC_STRIDE,  # int: stride for advancing batch in y: (C_TOT * H * W)
    C_OFFSET,     # int: channel offset inside destination y to start writing
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along NHW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along output channels (this branch)
    mask_m = offs_m < NHW
    mask_n = offs_n < C_OUT

    HW = H * W

    n_idx = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    # Flattened base offsets for x/y
    base_x = n_idx * (C_IN * HW) + h_idx * W + w_idx
    base_y = n_idx * Y_OC_STRIDE + h_idx * W + w_idx

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 1x1 conv: software-pipelined dot over input channels to hide load latency
    if C_IN > 0:
        x_vals = tl.load(x_ptr + (base_x + 0 * HW), mask=mask_m, other=0.0)
        w_vals = tl.load(w_ptr + (offs_n * C_IN + 0), mask=mask_n, other=0.0)
        for ci in tl.static_range(1, C_IN):
            acc += x_vals[:, None] * w_vals[None, :]
            x_vals = tl.load(x_ptr + (base_x + ci * HW), mask=mask_m, other=0.0)
            w_vals = tl.load(w_ptr + (offs_n * C_IN + ci), mask=mask_n, other=0.0)
        acc += x_vals[:, None] * w_vals[None, :]

    # Bias + ReLU
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store into destination with channel offset
    y_offs = base_y[:, None] + (C_OFFSET + offs_n)[None, :] * HW
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_offs, acc, mask=mask_store)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["NHW", "C_OUT"],
)
@triton.jit
def conv3x3_relu_kernel(
    x_ptr,        # *f32 [N, C_IN, H, W]
    w_ptr,        # *f32 [C_OUT, C_IN, 3, 3] contiguous
    b_ptr,        # *f32 [C_OUT]
    y_ptr,        # *f32 [N, C_TOT, H, W] (destination, may be larger than C_OUT)
    NHW,          # int: N * H * W
    H,            # int
    W,            # int
    Y_OC_STRIDE,  # int: stride for advancing batch in y: (C_TOT * H * W)
    C_OFFSET,     # int: channel offset inside destination y to start writing
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along NHW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along output channels (this branch)
    mask_m = offs_m < NHW
    mask_n = offs_n < C_OUT

    HW = H * W

    n_idx = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    base_y = n_idx * Y_OC_STRIDE + h_idx * W + w_idx

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 3x3 convolution with padding=1 over input channels
    for r in tl.static_range(0, 3):
        h_off = h_idx + (r - 1)
        h_ok = (h_off >= 0) & (h_off < H)
        for s in tl.static_range(0, 3):
            w_off = w_idx + (s - 1)
            w_ok = (w_off >= 0) & (w_off < W)
            in_mask = mask_m & h_ok & w_ok

            base_hw = h_off * W + w_off
            base_n = n_idx * (C_IN * HW) + base_hw

            if C_IN > 0:
                x_vals = tl.load(x_ptr + (base_n + 0 * HW), mask=in_mask, other=0.0)
                w_vals = tl.load(w_ptr + (offs_n * (C_IN * 9) + 0 * 9 + r * 3 + s),
                                 mask=mask_n, other=0.0)
                for ci in tl.static_range(1, C_IN):
                    acc += x_vals[:, None] * w_vals[None, :]
                    x_vals = tl.load(x_ptr + (base_n + ci * HW), mask=in_mask, other=0.0)
                    w_vals = tl.load(w_ptr + (offs_n * (C_IN * 9) + ci * 9 + r * 3 + s),
                                     mask=mask_n, other=0.0)
                acc += x_vals[:, None] * w_vals[None, :]

    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store into destination with channel offset
    y_offs = base_y[:, None] + (C_OFFSET + offs_n)[None, :] * HW
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_offs, acc, mask=mask_store)


# Store directly into a larger destination tensor with channel offset (avoid extra concat copies)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["NHW", "C_OUT"],
)
@triton.jit
def conv1x1_relu_store_cat_kernel(
    x_ptr,       # *f32 [N, C_IN, H, W]
    w_ptr,       # *f32 [C_OUT, C_IN, 1, 1]
    b_ptr,       # *f32 [C_OUT]
    dst_ptr,     # *f32 [N, C_OUT_TOTAL, H, W]
    NHW,         # int: N*H*W
    H,           # int
    W,           # int
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    C_OUT_TOTAL: tl.constexpr,
    C_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < NHW
    mask_n = offs_n < C_OUT

    HW = H * W

    n_idx = offs_m // HW
    hw_rem = offs_m % HW
    h_idx = hw_rem // W
    w_idx = hw_rem % W

    base_x = n_idx * (C_IN * HW) + h_idx * W + w_idx

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if C_IN > 0:
        x_vals = tl.load(x_ptr + (base_x + 0 * HW), mask=mask_m, other=0.0)
        w_vals = tl.load(w_ptr + (offs_n * C_IN + 0), mask=mask_n, other=0.0)
        for ci in tl.static_range(1, C_IN):
            acc += x_vals[:, None] * w_vals[None, :]
            x_vals = tl.load(x_ptr + (base_x + ci * HW), mask=mask_m, other=0.0)
            w_vals = tl.load(w_ptr + (offs_n * C_IN + ci), mask=mask_n, other=0.0)
        acc += x_vals[:, None] * w_vals[None, :]

    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = tl.maximum(acc + bias_vals[None, :], 0.0)

    # Store into destination with total output channels and offset
    base_y_total = n_idx * (C_OUT_TOTAL * HW) + h_idx * W + w_idx
    y_offs = base_y_total[:, None] + (C_OFFSET + offs_n)[None, :] * HW
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(dst_ptr + y_offs, acc, mask=mask_store)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["NHW", "C_OUT"],
)
@triton.jit
def conv3x3_relu_store_cat_kernel(
    x_ptr,       # *f32 [N, C_IN, H, W]
    w_ptr,       # *f32 [C_OUT, C_IN, 3, 3]
    b_ptr,       # *f32 [C_OUT]
    dst_ptr,     # *f32 [N, C_OUT_TOTAL, H, W]
    NHW,         # int: N*H*W
    H,           # int
    W,           # int
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    C_OUT_TOTAL: tl.constexpr,
    C_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < NHW
    mask_n = offs_n < C_OUT

    HW = H * W

    n_idx = offs_m // HW
    hw_rem = offs_m % HW
    h_idx = hw_rem // W
    w_idx = hw_rem % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 3x3 with padding=1
    for r in tl.static_range(0, 3):
        h_off = h_idx + (r - 1)
        h_ok = (h_off >= 0) & (h_off < H)
        for s in tl.static_range(0, 3):
            w_off = w_idx + (s - 1)
            w_ok = (w_off >= 0) & (w_off < W)
            in_mask = mask_m & h_ok & w_ok

            base_hw = h_off * W + w_off
            base_n = n_idx * (C_IN * HW) + base_hw

            if C_IN > 0:
                x_vals = tl.load(x_ptr + (base_n + 0 * HW), mask=in_mask, other=0.0)
                w_vals = tl.load(w_ptr + (offs_n * (C_IN * 9) + 0 * 9 + r * 3 + s),
                                 mask=mask_n, other=0.0)
                for ci in tl.static_range(1, C_IN):
                    acc += x_vals[:, None] * w_vals[None, :]
                    x_vals = tl.load(x_ptr + (base_n + ci * HW), mask=in_mask, other=0.0)
                    w_vals = tl.load(w_ptr + (offs_n * (C_IN * 9) + ci * 9 + r * 3 + s),
                                     mask=mask_n, other=0.0)
                acc += x_vals[:, None] * w_vals[None, :]

    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = tl.maximum(acc + bias_vals[None, :], 0.0)

    # Store into destination with total output channels and offset
    base_y_total = n_idx * (C_OUT_TOTAL * HW) + h_idx * W + w_idx
    y_offs = base_y_total[:, None] + (C_OFFSET + offs_n)[None, :] * HW
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(dst_ptr + y_offs, acc, mask=mask_store)


def _conv1x1_relu_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes y = ReLU(Conv2d1x1(x, w, b)) using Triton.
    x: [N, C_IN, H, W], w: [C_OUT, C_IN, 1, 1], b: [C_OUT]
    Returns y: [N, C_OUT, H, W]
    """
    assert x.ndim == 4 and w.ndim == 4
    N, C_IN, H, W = x.shape
    C_OUT = w.shape[0]
    x_c = x.contiguous()
    w_c = w.contiguous()
    b_c = torch.zeros(C_OUT, device=x.device, dtype=x.dtype) if b is None else b.contiguous()
    y = torch.empty((N, C_OUT, H, W), device=x.device, dtype=x.dtype)

    NHW = N * H * W
    HW = H * W
    Y_OC_STRIDE = C_OUT * HW
    # Grid: M dimension over NHW, N dimension over C_OUT
    def grid(META):
        return (triton.cdiv(NHW, META["BLOCK_M"]), triton.cdiv(C_OUT, META["BLOCK_N"]))
    conv1x1_relu_kernel[grid](
        x_c, w_c, b_c, y,
        NHW, H, W,
        Y_OC_STRIDE, 0,
        C_IN=C_IN, C_OUT=C_OUT,
    )
    return y


def _expand_relu_cat_triton(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
                            w3: torch.Tensor, b3: torch.Tensor) -> torch.Tensor:
    """
    Fused expand branches: computes
      y1 = ReLU(Conv1x1(x, w1, b1))
      y3 = ReLU(Conv3x3(x, w3, b3, padding=1))
    and writes them directly concatenated along channel dimension into a single output.
    x: [N, C_IN, H, W], w1: [C1, C_IN, 1, 1], w3: [C3, C_IN, 3, 3]
    returns out: [N, C1+C3, H, W]
    """
    N, C_IN, H, W = x.shape
    C1 = w1.shape[0]
    C3 = w3.shape[0]
    x_c = x.contiguous()
    w1_c = w1.contiguous()
    w3_c = w3.contiguous()
    b1_c = torch.zeros(C1, device=x.device, dtype=x.dtype) if b1 is None else b1.contiguous()
    b3_c = torch.zeros(C3, device=x.device, dtype=x.dtype) if b3 is None else b3.contiguous()

    out = torch.empty((N, C1 + C3, H, W), device=x.device, dtype=x.dtype)

    NHW = N * H * W
    HW = H * W
    Y_OC_STRIDE = (C1 + C3) * HW

    # Launch Conv1x1 -> write to channels [0..C1-1]
    def grid1(META):
        return (triton.cdiv(NHW, META["BLOCK_M"]), triton.cdiv(C1, META["BLOCK_N"]))
    conv1x1_relu_store_cat_kernel[grid1](
        x_c, w1_c, b1_c, out,
        NHW, H, W,
        C_IN=C_IN, C_OUT=C1,
        C_OUT_TOTAL=(C1 + C3), C_OFFSET=0,
    )

    # Launch Conv3x3 -> write to channels [C1..C1+C3-1]
    def grid3(META):
        return (triton.cdiv(NHW, META["BLOCK_M"]), triton.cdiv(C3, META["BLOCK_N"]))
    conv3x3_relu_store_cat_kernel[grid3](
        x_c, w3_c, b3_c, out,
        NHW, H, W,
        C_IN=C_IN, C_OUT=C3,
        C_OUT_TOTAL=(C1 + C3), C_OFFSET=C1,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(ModelNew, self).__init__()

        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        if x.is_cuda:
            # Squeeze 1x1 + ReLU (Triton)
            x_s = _conv1x1_relu_triton(x, self.squeeze.weight, self.squeeze.bias)
            # Fused expand branches into a single concatenated output (Triton)
            out = _expand_relu_cat_triton(x_s, self.expand1x1.weight, self.expand1x1.bias,
                                          self.expand3x3.weight, self.expand3x3.bias)
            return out
        else:
            # Fallback to exact original semantics on CPU
            x = self.squeeze_activation(self.squeeze(x))
            return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ], 1)


# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width, device='cuda')]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]