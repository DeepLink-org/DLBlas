import torch
import torch.nn as nn

# Try to import Triton. If unavailable, we'll fall back to PyTorch ops.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _sigmoid_affine_kernel(
        x_ptr,          # *f32
        scale_ptr,      # *f32, shape [1]
        base_ptr,       # *f32, shape [K]
        y_ptr,          # *f32
        total_elems,    # i32
        K,              # i32, last dimension
        eps,            # f32
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_elems

        # Load input
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # Load scale (broadcast scalar)
        s = tl.load(scale_ptr)

        # Broadcast base over last dimension using modulo
        col = offs % K
        base = tl.load(base_ptr + col, mask=mask, other=0.0)

        # Affine transform + sigmoid + eps
        y = x * s + base
        y = tl.sigmoid(y)
        y = y + eps

        # Store
        tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Implements:
        output = torch.sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_pre_eps: float,
    ) -> torch.Tensor:
        # Fallback to PyTorch if Triton unavailable or running on CPU


        assert input_mix.dim() == 3, "input_mix must be 3D [B, N, C]"
        B, N, C = input_mix.shape
        # Ensure base matches last dimension
        assert mhc_base.numel() == C, "mhc_base must have same size as last dim of input_mix"

        # Make tensors contiguous and on same dtype/device
        x = input_mix.contiguous()
        dtype = x.dtype
        device = x.device
        scale = mhc_scale.to(device=device, dtype=dtype).contiguous()
        base = mhc_base.to(device=device, dtype=dtype).contiguous()

        # Allocate output tensor
        y = torch.empty_like(x)

        # Flatten over all elements for simple coalesced access
        total = x.numel()
        BLOCK = 4096
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)

        _sigmoid_affine_kernel[grid](
            x, scale, base, y,
            total, C, mhc_pre_eps.float(),
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return y


batch_size = 16
n1 = 16384
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch_size, n1, mhc_mult)
    mhc_scale = torch.randn(1)
    mhc_base = torch.randn(mhc_mult)
    mhc_pre_eps = 1e-2
    return [input_mix, mhc_scale, mhc_base, mhc_pre_eps]


def get_init_inputs():
    return []