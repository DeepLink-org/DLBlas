import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, x, group_size, eps=1e-10, dtype=None, scale_ue8m0=False
    ):
        if dtype is None:
            dtype = torch.float8_e4m3fn  # flag_gems.SUPPORTED_FP8_DTYPE

        assert (
            x.shape[-1] % group_size == 0
        ), "the last dimension of `x` cannot be divisible by `group_size`"
        assert x.is_contiguous(), "`x` is not contiguous"

        finfo = torch.finfo(dtype)
        fp8_min = finfo.min
        fp8_max = finfo.max

        x_ = x.reshape(x.numel() // group_size, group_size)
        amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
        x_s = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=x.device)
        if scale_ue8m0:
            min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
            x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
        x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
        x_q = x_q.reshape(x.shape)
        x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

        return x_q, x_s


def get_inputs():
    num_tokens = 7
    d = 512
    group_size = 512
    dtype = torch.bfloat16
    x = torch.rand(num_tokens, d, dtype=dtype, device='cuda')

    return [x, group_size]


def get_init_inputs():
    return []
