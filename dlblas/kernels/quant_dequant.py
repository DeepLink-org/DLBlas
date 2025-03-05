import torch
from torchao.float8.float8_utils import (
    to_fp8_saturated,
)

def per_channel_quant_fp8(x, quant_dtype=torch.float8_e4m3fn):
    x_amax = x.abs().amax(-1, True)
    x_scales = x_amax.float() / torch.finfo(quant_dtype).max
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    return x_quanted, x_scales

def per_channel_dequant_bf16(x_quanted, x_scales, dequant_dtype=torch.bfloat16):
    x_origin = x_quanted.float() * x_scales
    return x_origin.to(dequant_dtype)
