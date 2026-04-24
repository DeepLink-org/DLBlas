
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
        torch.backends.cuda.matmul.allow_tf32 = True
        if mhc_norm_weight is not None:
            mhc_fn = mhc_fn * mhc_norm_weight
        residual = residual.flatten(2, 3).float()
        assert mhc_fn.dtype == residual.dtype == torch.float
        mhc_mult = mhc_fn.shape[0]
        rms_group_size = mhc_fn.shape[-1]
        mixes = torch.einsum(
            'mbk,nbk->mbn',
            residual.view(-1, 1, rms_group_size),
            mhc_fn.view(mhc_mult, 1, rms_group_size),
        )
        sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
        mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
        torch.backends.cuda.matmul.allow_tf32 = False
        return mixes.view(*residual.shape[:2], -1)

n1=13
mhc_mult=4
hidden_size=1280
generate_normw=False

def generate_norm_fn_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float)
        .mul(1 + torch.arange(mhc_mult).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(mhc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    if generate_normw:
        normw = torch.randn((mhc_hidden_size,), dtype=torch.float) * 0.1 + 1.0
    else:
        normw = None

    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float)

    return [residual,fn,normw,out_grad,1e-6]


def get_inputs():
    residual, fn, normw, out_grad, mhc_norm_eps = generate_norm_fn_test_data(n1, mhc_mult, hidden_size, generate_normw)
    return [residual, fn, None, mhc_norm_eps]

def get_init_inputs():
    return []
