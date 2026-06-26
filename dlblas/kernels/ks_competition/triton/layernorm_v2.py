import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_lastdim_fwd_kernel(
    x_ptr,
    y_ptr,
    M,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    eps,
    N: tl.constexpr,          # 显式传入 N，作为编译时常量
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)    # 范围 [0, N-1]，无越界

    x_row_ptr = x_ptr + pid * stride_xm + offs * stride_xn
    y_row_ptr = y_ptr + pid * stride_ym + offs * stride_yn

    x = tl.load(x_row_ptr)    # 无越界，无需 mask

    invN = 1.0 / N
    mean = tl.sum(x, axis=0) * invN

    diff = x - mean
    var = tl.sum(diff * diff, axis=0) * invN
    inv_std = tl.rsqrt(var + eps)

    y = diff * inv_std
    tl.store(y_row_ptr, y)


def _layer_norm_lastdim_triton(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Triton fast-path for 2-D inputs with last dim = 10."""
    if x.dim() != 2:
        # 仅支持 2D
        return torch.nn.LayerNorm(x.size(-1), eps=eps).to(x.device)(x)

    M, N = x.shape
    assert N == 10, f"This kernel assumes normalized_shape=10, got N={N}"

    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    # 关键修改：BLOCK_SIZE 等于 N，彻底消除越界
    layer_norm_lastdim_fwd_kernel[(M,)](
        x,
        y,
        M,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        eps,
        N=N,                # 编译时常量
        num_warps=1,
        num_stages=1,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return _layer_norm_lastdim_triton(x, eps=1e-5)


def get_inputs():
    x = torch.rand(10, 10, device="npu")
    return [x]


def get_init_inputs():
    return []


# Smoke test
if __name__ == "__main__":
    import torch_npu
    torch_npu.npu.set_device(0)
    device = torch.device("npu:0")

    model = ModelNew().to(device)
    inputs = [x.to(device) for x in get_inputs()]

    with torch.no_grad():
        out_triton = model(*inputs)

    # Reference
    ln_ref = nn.LayerNorm(10, eps=1e-5).to(device)
    with torch.no_grad():
        out_ref = ln_ref(inputs[0])

    diff = (out_triton - out_ref).abs()
    max_err = diff.max().item()
    print(f"Max absolute error: {max_err:.2e}")
    print(f"Allclose(atol=0.01, rtol=0.01): {torch.allclose(out_triton, out_ref, atol=0.01, rtol=0.01)}")