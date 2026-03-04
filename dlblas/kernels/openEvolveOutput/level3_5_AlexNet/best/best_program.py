# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_relu(
    x_ptr, w_ptr, b_ptr, output_ptr,
    in_features, out_features,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < out_features
    
    # Load weights and bias
    w = tl.load(w_ptr + offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + offset, mask=mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute dot product
    for k in range(0, in_features, BLOCK_SIZE):
        cols = k + tl.arange(0, BLOCK_SIZE)
        x_mask = cols < in_features
        x_val = tl.load(x_ptr + cols, mask=x_mask, other=0.0)
        
        w_cols = offset + cols * out_features
        w_mask = (cols < in_features) & mask
        w_val = tl.load(w_ptr + w_cols, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_val, w_val)
    
    # Add bias and apply ReLU
    result = acc + b
    if ACTIVATION:
        result = tl.where(result > 0, result, 0.0)
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Convolutional layers (unchanged)
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        
        # Activation and pooling layers (unchanged)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        
        # Fully connected layers with Triton optimized implementation
        self.fc1_weight = nn.Parameter(torch.Tensor(4096, 256*6*6))
        self.fc1_bias = nn.Parameter(torch.Tensor(4096))
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc1_weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1_bias, 0)
    
    def forward(self, x):
        # Convolutional layers (unchanged)
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(self.relu(self.conv5(x)))
        
        # Flatten and fully connected layer with Triton optimization
        x = torch.flatten(x, 1)
        
        # Triton-optimized fused linear + ReLU
        out_features = self.fc1_weight.size(0)
        in_features = self.fc1_weight.size(1)
        output = torch.empty(x.size(0), out_features, device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE = 128
        grid = (triton.cdiv(out_features, BLOCK_SIZE),)
        fused_linear_relu[grid](
            x, self.fc1_weight, self.fc1_bias, output,
            in_features, out_features,
            BLOCK_SIZE=BLOCK_SIZE,
            ACTIVATION=True
        )
        
        # Remaining layers
        x = self.relu(output)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================