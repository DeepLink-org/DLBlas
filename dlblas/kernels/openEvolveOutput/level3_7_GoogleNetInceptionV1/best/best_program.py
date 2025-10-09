# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, batch_size, height, width,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_w0, stride_w1,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_bhw = tl.program_id(0)
    pid_oc = tl.program_id(1)
    
    num_bhw = batch_size * height * width
    b_idx = pid_bhw // (height * width)
    hw_idx = pid_bhw % (height * width)
    h_idx = hw_idx // width
    w_idx = hw_idx % width
    
    if b_idx >= batch_size or h_idx >= height or w_idx >= width:
        return
    
    oc_block = tl.arange(0, BLOCK_C)
    oc_mask = oc_block < (out_channels - pid_oc * BLOCK_C)
    
    output_offsets = (
        b_idx * stride_ob + 
        (pid_oc * BLOCK_C + oc_block) * stride_oc +
        h_idx * stride_oh +
        w_idx * stride_ow
    )
    
    bias_offsets = pid_oc * BLOCK_C + oc_block
    acc = tl.load(bias_ptr + bias_offsets, mask=oc_mask, other=0.0)
    
    for ic in range(0, in_channels, BLOCK_K):
        ic_offsets = ic + tl.arange(0, BLOCK_K)
        ic_mask = ic_offsets < in_channels
        
        x_offsets = (
            b_idx * stride_xb +
            ic_offsets * stride_xc +
            h_idx * stride_xh +
            w_idx * stride_xw
        )
        
        w_offsets = (
            (pid_oc * BLOCK_C + oc_block)[:, None] * stride_w0 +
            ic_offsets[None, :] * stride_w1
        )
        
        x = tl.load(input_ptr + x_offsets, mask=ic_mask, other=0.0)
        w = tl.load(weight_ptr + w_offsets, mask=ic_mask[None, :] & oc_mask[:, None], other=0.0)
        
        acc += tl.sum(w * x, axis=1)
    
    tl.store(output_ptr + output_offsets, acc, mask=oc_mask)

def triton_conv1x1(x, weight, bias):
    batch, in_c, h, w = x.shape
    out_c = weight.shape[0]
    output = torch.empty(batch, out_c, h, w, device=x.device, dtype=x.dtype)
    
    grid = (batch * h * w, triton.cdiv(out_c, 64))
    BLOCK_C = 64
    BLOCK_K = 64
    
    conv1x1_kernel[grid](
        x, weight, bias, output,
        in_c, out_c, batch, h, w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K,
        num_warps=4
    )
    return output

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = triton_conv1x1(x, 
            self.branch1x1.weight.view(self.branch1x1.weight.size(0), -1), 
            self.branch1x1.bias
        )
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================