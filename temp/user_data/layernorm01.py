import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(10)

    def forward(self, x):
        if self.ln.weight is not None:
            self.ln.weight.data = self.ln.weight.data.to(x.device)
        if self.ln.bias is not None:
            self.ln.bias.data = self.ln.bias.data.to(x.device)
        return self.ln(x)

def get_inputs():
    x = torch.rand(10, 10)
    return [x]

def get_init_inputs():
    return []

if __name__ == "__main__":
      import torch_npu
      torch_npu.npu.set_device(0)
      device = torch.device("npu:0")

      model = Model().to(device)
      inputs = [x.to(device) for x in get_inputs()]
      with torch.no_grad():
          res = model(*inputs)
      print(res.shape)