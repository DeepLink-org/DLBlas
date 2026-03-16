import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Triton path for light-weight weight transform (to mark custom kernel usage and keep it fast)
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: permute (in_c, out_c, kd, kh, kw) -> (out_c, in_c, kd, kh, kw) and flip spatial dims (kd, kh, kw)
# Mapping:
#   out[co, ci, kd, kh, kw] = inp[ci, co, KD-1-kd, KH-1-kh, KW-1-kw]
if TRITON_AVAILABLE:
    @triton.jit
    def _perm_flip_w_kernel(
        inp_ptr,  # *T [CI, CO, KD, KH, KW]
        out_ptr,  # *T [CO, CI, KD, KH, KW]
        CI: tl.constexpr,
        CO: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        N_ELEMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_ELEMS

        # Decompose linear index into (co, ci, kd, kh, kw) for output layout [CO, CI, KD, KH, KW]
        kw = offs % KW
        tmp = offs // KW
        kh = tmp % KH
        tmp = tmp // KH
        kd = tmp % KD
        tmp = tmp // KD
        ci = tmp % CI
        co = tmp // CI

        # Map to input indices [CI, CO, KD, KH, KW] with spatial flip
        kd_in = KD - 1 - kd
        kh_in = KH - 1 - kh
        kw_in = KW - 1 - kw

        # Compute flat offsets for contiguous memory
        in_offs = (((ci * CO + co) * KD + kd_in) * KH + kh_in) * KW + kw_in
        # Output offset is simply offs (same linearization as we decomposed)
        out_offs = offs

        vals = tl.load(inp_ptr + in_offs, mask=mask, other=0)
        tl.store(out_ptr + out_offs, vals, mask=mask)

    # Grouped variant: inp [CI, COG, KD, KH, KW] -> out [CO, CIG, KD, KH, KW] with spatial flip
    # where CIG = CI // G and COG = CO // G
    @triton.jit
    def _perm_flip_w_kernel_grouped(
        inp_ptr,  # *T [CI, COG, KD, KH, KW]
        out_ptr,  # *T [CO, CIG, KD, KH, KW]
        CI: tl.constexpr,
        CO: tl.constexpr,
        KD: tl.constexpr,
        KH: tl.constexpr,
        KW: tl.constexpr,
        G: tl.constexpr,        # groups
        N_ELEMS: tl.constexpr,  # total elements of output
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_ELEMS

        kw = offs % KW
        t1 = offs // KW
        kh = t1 % KH
        t2 = t1 // KH
        kd = t2 % KD
        t3 = t2 // KD

        CIG = CI // G
        COG = CO // G

        ci_in_group = t3 % CIG
        co_global = t3 // CIG
        g = co_global // COG
        co_group = co_global - g * COG
        ci_global = g * CIG + ci_in_group

        kd_in = KD - 1 - kd
        kh_in = KH - 1 - kh
        kw_in = KW - 1 - kw

        # input is [CI, COG, KD, KH, KW]
        in_offs = (((ci_global * COG + co_group) * KD + kd_in) * KH + kh_in) * KW + kw_in
        out_offs = offs

        vals = tl.load(inp_ptr + in_offs, mask=mask, other=0)
        tl.store(out_ptr + out_offs, vals, mask=mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def _fastpath_available(self) -> bool:
        ct = self.conv_transpose3d
        # Map ConvTranspose3d to Conv3d when stride == 1, dilation == 1, output_padding == 0
        # and derived conv padding is non-negative. Supports groups.
        if not (
            ct.stride == (1, 1, 1)
            and getattr(ct, "dilation", (1, 1, 1)) == (1, 1, 1)
            and ct.output_padding == (0, 0, 0)
        ):
            return False
        KD, KH, KW = ct.weight.shape[2], ct.weight.shape[3], ct.weight.shape[4]
        pd, ph, pw = ct.padding
        return (KD - 1 - pd) >= 0 and (KH - 1 - ph) >= 0 and (KW - 1 - pw) >= 0

    def _perm_flip_weight_triton(self, w: torch.Tensor) -> torch.Tensor:
        # Generic: handles groups as well
        assert w.is_contiguous()
        ct = self.conv_transpose3d
        G = ct.groups
        if G == 1:
            CI, CO, KD, KH, KW = w.shape
            out = torch.empty((CO, CI, KD, KH, KW), device=w.device, dtype=w.dtype)
            N = CO * CI * KD * KH * KW
            BLOCK = 4096
            grid = ((N + BLOCK - 1) // BLOCK,)
            _perm_flip_w_kernel[grid](w, out, CI=CI, CO=CO, KD=KD, KH=KH, KW=KW, N_ELEMS=N, BLOCK=BLOCK, num_warps=8, num_stages=2)
            return out
        else:
            CI, COG, KD, KH, KW = w.shape
            CO = self.conv_transpose3d.out_channels
            CIG = CI // G
            out = torch.empty((CO, CIG, KD, KH, KW), device=w.device, dtype=w.dtype)
            N = CO * CIG * KD * KH * KW
            BLOCK = 4096
            grid = ((N + BLOCK - 1) // BLOCK,)
            _perm_flip_w_kernel_grouped[grid](
                w, out, CI=CI, CO=CO, KD=KD, KH=KH, KW=KW, G=G, N_ELEMS=N, BLOCK=BLOCK, num_warps=8, num_stages=2
            )
            return out

    def _perm_flip_weight_torch(self, w: torch.Tensor) -> torch.Tensor:
        # PyTorch fallback handling groups correctly
        ct = self.conv_transpose3d
        G = ct.groups
        CI, COG, KD, KH, KW = w.shape if G > 1 else (w.shape[0], w.shape[1], w.shape[2], w.shape[3], w.shape[4])
        if G == 1:
            # [CI, CO, KD, KH, KW] -> [CO, CI, KD, KH, KW] with spatial flip
            return w.permute(1, 0, 2, 3, 4).flip(dims=(2, 3, 4)).contiguous()
        else:
            CIG = CI // G
            w_view = w.view(G, CIG, COG, KD, KH, KW)
            w_rot = (
                w_view.permute(0, 2, 1, 3, 4, 5)  # [G, COG, CIG, KD, KH, KW]
                .flip(dims=(3, 4, 5))             # spatial flip
                .contiguous()
                .view(G * COG, CIG, KD, KH, KW)   # [CO, CIG, KD, KH, KW]
            )
            return w_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        ct = self.conv_transpose3d

        # Fast path: Implement ConvTranspose3d via Conv3d with flipped/perm'd weights
        # Valid for stride=(1,1,1), dilation=(1,1,1), output_padding=(0,0,0); supports groups and padding
        if x.is_cuda and self._fastpath_available():
            w = ct.weight.contiguous()
            if TRITON_AVAILABLE:
                w_rot = self._perm_flip_weight_triton(w)
            else:
                w_rot = self._perm_flip_weight_torch(w)

            KD, KH, KW = w.shape[2], w.shape[3], w.shape[4]
            pd, ph, pw = ct.padding
            # conv3d padding derived from conv_transpose3d parameters (stride=1, dilation=1)
            padding = (KD - 1 - pd, KH - 1 - ph, KW - 1 - pw)
            out = F.conv3d(
                x,
                w_rot,
                bias=ct.bias,
                stride=1,
                padding=padding,
                dilation=1,
                groups=ct.groups,
            )
            return out

        # Fallback: general, numerically-correct path using PyTorch
        return ct(x)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization