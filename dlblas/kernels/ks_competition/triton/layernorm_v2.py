import torch
import torch_npu
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit(noinline=True)
def layer_norm_lastdim_affine_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    stride_row_x, stride_col_x,
    stride_col_w, stride_col_b,
    N_COLS, eps, BLOCK_SIZE: tl.constexpr,
):
    """
    1D grid — one program per row.  BLOCK_SIZE == N_COLS → zero masking.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)          # [0 … N_COLS-1], all valid

    row_start = pid * stride_row_x

    # ── Load N_COLS elements (no mask, no other=) ─────────────────────
    x_ptrs = X_ptr + row_start + offs * stride_col_x
    x = tl.load(x_ptrs).to(tl.float32)

    # ── Mean ──────────────────────────────────────────────────────────
    sum_x = tl.sum(x)
    mean = sum_x / N_COLS

    # ── Variance: E[(x - mean)²] ──────────────────────────────────────
    diff = x - mean
    var = tl.sum(diff * diff) / N_COLS

    # ── Normalise ─────────────────────────────────────────────────────
    rstd = tl.rsqrt(var + eps)
    x_norm = diff * rstd

    # ── Affine ────────────────────────────────────────────────────────
    w_ptrs = W_ptr + offs * stride_col_w
    b_ptrs = B_ptr + offs * stride_col_b
    w = tl.load(w_ptrs).to(tl.float32)
    b = tl.load(b_ptrs).to(tl.float32)

    y_f32 = x_norm * w + b
    y = y_f32.to(x.dtype)

    # ── Store ─────────────────────────────────────────────────────────
    y_ptrs = Y_ptr + row_start + offs * stride_col_x
    tl.store(y_ptrs, y)


class ModelNew(nn.Module):
    def __init__(self, num_features=10, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.block_size = num_features            # exact match → no masking

        self.weight = nn.Parameter(torch.ones(num_features, device="npu"))
        self.bias = nn.Parameter(torch.zeros(num_features, device="npu"))

    def forward(self, x):
        x_in = x.contiguous()
        N, C = x_in.shape
        assert C == self.num_features
        y = torch.empty(x_in.shape, dtype=x_in.dtype, device=x_in.device)

        stride_row_x = x_in.stride(0)
        stride_col_x = x_in.stride(1)
        w = self.weight
        b = self.bias
        stride_col_w = w.stride(0)
        stride_col_b = b.stride(0)

        grid = (N,)
        layer_norm_lastdim_affine_kernel[grid](
            x_in, w, b, y,
            stride_row_x, stride_col_x,
            stride_col_w, stride_col_b,
            C, self.eps,
            BLOCK_SIZE=self.block_size,
            num_warps=1
        )
        return y


def get_inputs():
    torch.manual_seed(42)
    x = torch.rand(10, 10).npu()
    return [x]


def get_init_inputs():
    return []


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    dev = torch.device("npu:0")
    model = ModelNew(num_features=10).to(dev)
    inputs = get_inputs()
    out_triton = model(*inputs)

    ln_ref = nn.LayerNorm(10, eps=1e-5).to(dev)
    with torch.no_grad():
        ln_ref.weight.copy_(model.weight)
        ln_ref.bias.copy_(model.bias)
    out_ref = ln_ref(inputs[0])

    diff = (out_triton - out_ref).abs()
    max_err = diff.max().item()
    print(f"Triton LN shape: {out_triton.shape}")
    print(f"Max abs error: {max_err:.2e}")
    print(f"Pass tolerance: {torch.allclose(out_triton, out_ref, atol=0.01, rtol=0.01)}")