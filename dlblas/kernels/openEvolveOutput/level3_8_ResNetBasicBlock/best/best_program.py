# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv_bn_relu_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    stride, padding,
    in_channels, out_channels,
    H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_operations = out_channels * H * W
    operations_per_pid = (num_operations + num_pid - 1) // num_pid
    
    start = pid * operations_per_pid
    end = tl.minimum(start + operations_per_pid, num_operations)
    
    for op_idx in range(start, end):
        c = op_idx // (H * W)
        hw = op_idx % (H * W)
        h = hw // W
        w = hw % W
        
        acc = 0.0
        for kh in range(3):
            for kw in range(3):
                h_in = h * stride - padding + kh
                w_in = w * stride - padding + kw
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    for ic in range(in_channels):
                        x_offset = ic * stride_xc + h_in * stride_xh + w_in * stride_xw
                        w_offset = c * stride_wc + ic * stride_wn + kh * stride_wh + kw * stride_ww
                        x_val = tl.load(x_ptr + x_offset)
                        w_val = tl.load(w_ptr + w_offset)
                        acc += x_val * w_val
        
        bias_val = tl.load(b_ptr + c)
        acc += bias_val
        acc = tl.maximum(acc, 0.0)
        out_offset = c * stride_oc + h * stride_oh + w * stride_ow
        tl.store(output_ptr + out_offset, acc)

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        self.register_buffer('fused_weight', torch.empty_like(self.conv1.weight))
        self.register_buffer('fused_bias', torch.empty(out_channels))

    def fuse_conv_bn(self):
        with torch.no_grad():
            conv_weight = self.conv1.weight
            bn_weight = self.bn1.weight
            bn_bias = self.bn1.bias
            bn_mean = self.bn1.running_mean
            bn_var = self.bn1.running_var
            bn_eps = self.bn1.eps
            
            scale_factor = bn_weight / torch.sqrt(bn_var + bn_eps)
            self.fused_weight.copy_(conv_weight * scale_factor[:, None, None, None])
            self.fused_bias.copy_(bn_bias - bn_mean * scale_factor)

    def forward(self, x):
        identity = x

        if self.training:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
        else:
            if self.fused_weight is None:
                self.fuse_conv_bn()
                
            B, C, H, W = x.shape
            output = torch.empty(B, self.conv1.out_channels, H // self.stride, W // self.stride, 
                                device=x.device, dtype=x.dtype)
            
            stride_x = x.stride()
            stride_w = self.fused_weight.stride()
            stride_o = output.stride()
            
            for b in range(B):
                conv_bn_relu_kernel[(1024,)](x[b], self.fused_weight, self.fused_bias, output[b],
                    self.stride, 1,
                    self.conv1.in_channels, self.conv1.out_channels,
                    output.size(2), output.size(3),
                    stride_x[1], stride_x[2], stride_x[3],
                    stride_w[1], stride_w[2], stride_w[3],
                    stride_o[1], stride_o[2], stride_o[3],
                    BLOCK_SIZE=128)
            out = output

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]
# =================== EVOLVE-BLOCK-END ===================