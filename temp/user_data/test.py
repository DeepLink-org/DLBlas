import torch
import torch_npu  # 必须放在使用torch.npu之前
import sys
print("cuda avail:", torch.cuda.is_available())
print("npu avail:", torch.npu.is_available())
print("Python路径:", sys.executable)