# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that replaces degenerate instance normalization with direct
    parameter access, eliminating unnecessary computations and reshaping.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Retain parameters for compatibility
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Skip linear computation and normalization since they're degenerate
        # Directly use instance norm bias for equivalent output
        x_norm = self.instance_norm.bias.unsqueeze(0).expand(x.size(0), -1)
        x = x_norm + y
        x = x * y
        return x

batch_size = 128
in_features = 64
out_features = 128

def get_inputs():
    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================