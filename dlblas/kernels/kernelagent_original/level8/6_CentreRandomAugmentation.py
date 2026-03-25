"""
Centre Random Augmentation (扩散采样里用于随机刚体变换)

From: protenix/model/utils.py:centre_random_augmentation
"""

import math
from typing import Optional
import torch
import torch.nn as nn

import triton
import triton.language as tl


def random_rotation_matrices(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    生成 n 个随机旋转矩阵 [n,3,3]，基于随机四元数（均匀分布）。
    """
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    # quaternion (x,y,z,w)
    x, y, z, w = q1, q2, q3, q4

    # convert to rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(n, 3, 3)
    return R


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    r: [...,3,3], t: [...,3]
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


@triton.jit
def _rot_trans_kernel(
    out_ptr,        # float*  [n_sample, N, 3]
    x_ptr,          # float*  [N, 3]   (centered input)
    R_ptr,          # float*  [n_sample, 3, 3]
    T_ptr,          # float*  [n_sample, 3]
    mask_ptr,       # float*  [N] or unused
    N,              # int     number of atoms
    out_s0, out_s1, out_s2,   # strides for out in elements
    x_s0, x_s1,               # strides for x in elements
    R_s0, R_s1, R_s2,         # strides for R in elements
    T_s0, T_s1,               # strides for T in elements
    mask_s0,                  # stride for mask in elements
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_s = tl.program_id(axis=0)  # sample id
    pid_b = tl.program_id(axis=1)  # block id over atoms

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < N
    tl.multiple_of(offs, BLOCK)

    # Load x columns individually
    base_x = offs * x_s0
    X = tl.load(x_ptr + base_x + 0 * x_s1, mask=in_bounds, other=0.0)
    Y = tl.load(x_ptr + base_x + 1 * x_s1, mask=in_bounds, other=0.0)
    Z = tl.load(x_ptr + base_x + 2 * x_s1, mask=in_bounds, other=0.0)

    # Load R for this sample (3x3)
    r00 = tl.load(R_ptr + pid_s * R_s0 + 0 * R_s1 + 0 * R_s2)
    r01 = tl.load(R_ptr + pid_s * R_s0 + 0 * R_s1 + 1 * R_s2)
    r02 = tl.load(R_ptr + pid_s * R_s0 + 0 * R_s1 + 2 * R_s2)
    r10 = tl.load(R_ptr + pid_s * R_s0 + 1 * R_s1 + 0 * R_s2)
    r11 = tl.load(R_ptr + pid_s * R_s0 + 1 * R_s1 + 1 * R_s2)
    r12 = tl.load(R_ptr + pid_s * R_s0 + 1 * R_s1 + 2 * R_s2)
    r20 = tl.load(R_ptr + pid_s * R_s0 + 2 * R_s1 + 0 * R_s2)
    r21 = tl.load(R_ptr + pid_s * R_s0 + 2 * R_s1 + 1 * R_s2)
    r22 = tl.load(R_ptr + pid_s * R_s0 + 2 * R_s1 + 2 * R_s2)

    # Load T for this sample (3)
    t0 = tl.load(T_ptr + pid_s * T_s0 + 0 * T_s1)
    t1 = tl.load(T_ptr + pid_s * T_s0 + 1 * T_s1)
    t2 = tl.load(T_ptr + pid_s * T_s0 + 2 * T_s1)

    # Compute y = R @ x + T using FMA chains
    y0 = tl.math.fma(r00, X, r01 * Y)
    y0 = tl.math.fma(r02, Z, y0) + t0
    y1 = tl.math.fma(r10, X, r11 * Y)
    y1 = tl.math.fma(r12, Z, y1) + t1
    y2 = tl.math.fma(r20, X, r21 * Y)
    y2 = tl.math.fma(r22, Z, y2) + t2

    if HAS_MASK:
        mvals = tl.load(mask_ptr + offs * mask_s0, mask=in_bounds, other=0.0)
        y0 = y0 * mvals
        y1 = y1 * mvals
        y2 = y2 * mvals

    # Store results to out: [n_sample, N, 3]
    out_base = out_ptr + pid_s * out_s0
    tl.store(out_base + offs * out_s1 + 0 * out_s2, y0, mask=in_bounds)
    tl.store(out_base + offs * out_s1 + 1 * out_s2, y1, mask=in_bounds)
    tl.store(out_base + offs * out_s1 + 2 * out_s2, y2, mask=in_bounds)


@triton.jit
def _rot_trans_kernel_contig(
    out_ptr,        # float*  [n_sample, N, 3] contiguous
    x_ptr,          # float*  [N, 3] contiguous
    R_ptr,          # float*  [n_sample, 3, 3] contiguous
    T_ptr,          # float*  [n_sample, 3] contiguous
    mask_ptr,       # float*  [N] contiguous or unused
    N,              # int     number of atoms
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_s = tl.program_id(axis=0)  # sample id
    pid_b = tl.program_id(axis=1)  # block id over atoms

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < N
    tl.multiple_of(offs, BLOCK)

    # Load centered coordinates x: three components (contiguous [N,3] -> stride (3,1))
    base_x = offs * 3
    X = tl.load(x_ptr + base_x + 0, mask=in_bounds, other=0.0)
    Y = tl.load(x_ptr + base_x + 1, mask=in_bounds, other=0.0)
    Z = tl.load(x_ptr + base_x + 2, mask=in_bounds, other=0.0)

    # Load rotation matrix for this sample (row-major 3x3) from contiguous R
    base_r = pid_s * 9
    r00 = tl.load(R_ptr + base_r + 0)
    r01 = tl.load(R_ptr + base_r + 1)
    r02 = tl.load(R_ptr + base_r + 2)
    r10 = tl.load(R_ptr + base_r + 3)
    r11 = tl.load(R_ptr + base_r + 4)
    r12 = tl.load(R_ptr + base_r + 5)
    r20 = tl.load(R_ptr + base_r + 6)
    r21 = tl.load(R_ptr + base_r + 7)
    r22 = tl.load(R_ptr + base_r + 8)

    # Load translation for this sample
    base_t = pid_s * 3
    t0 = tl.load(T_ptr + base_t + 0)
    t1 = tl.load(T_ptr + base_t + 1)
    t2 = tl.load(T_ptr + base_t + 2)

    # Apply rotation and translation: y = R @ x + T using FMA chains
    y0 = tl.math.fma(r00, X, r01 * Y)
    y0 = tl.math.fma(r02, Z, y0) + t0
    y1 = tl.math.fma(r10, X, r11 * Y)
    y1 = tl.math.fma(r12, Z, y1) + t1
    y2 = tl.math.fma(r20, X, r21 * Y)
    y2 = tl.math.fma(r22, Z, y2) + t2

    if HAS_MASK:
        w = tl.load(mask_ptr + offs, mask=in_bounds, other=0.0)
        y0 = y0 * w
        y1 = y1 * w
        y2 = y2 * w

    # Store results to out: [n_sample, N, 3] contiguous
    base_out = pid_s * (N * 3) + offs * 3
    tl.store(out_ptr + base_out + 0, y0, mask=in_bounds)
    tl.store(out_ptr + base_out + 1, y1, mask=in_bounds)
    tl.store(out_ptr + base_out + 2, y2, mask=in_bounds)


def _apply_rot_trans_triton(x_centered: torch.Tensor,
                            R: torch.Tensor,
                            T: torch.Tensor,
                            mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x_centered: [N, 3]
    R: [n, 3, 3]
    T: [n, 3]
    mask: [N] or None
    returns out: [n, N, 3]
    """
    assert x_centered.is_cuda and R.is_cuda and T.is_cuda
    device = x_centered.device
    dtype = x_centered.dtype
    n_sample = R.shape[0]
    N = x_centered.shape[0]

    out = torch.empty((n_sample, N, 3), device=device, dtype=dtype)

    # Prepare strides (in elements)
    out_s0, out_s1, out_s2 = out.stride()
    x_s0, x_s1 = x_centered.stride()
    R_s0, R_s1, R_s2 = R.stride()
    T_s0, T_s1 = T.stride()

    HAS_MASK = mask is not None
    if HAS_MASK:
        # Convert mask to output dtype as in the reference code
        mask_typed = mask.to(dtype=dtype)
        mask_s0 = mask_typed.stride(0)
        mask_ptr = mask_typed
    else:
        # Dummy tensor to satisfy pointer argument; will be ignored in kernel
        mask_typed = torch.empty(1, device=device, dtype=dtype)
        mask_s0 = 0
        mask_ptr = mask_typed

    BLOCK = 128
    grid = (n_sample, triton.cdiv(N, BLOCK))

    # Fast path for fully-contiguous layouts
    if (
        x_centered.is_contiguous()
        and out.is_contiguous()
        and R.is_contiguous()
        and T.is_contiguous()
        and (not HAS_MASK or mask_typed.is_contiguous())
        and x_s0 == 3 and x_s1 == 1
        and R_s0 == 9 and R_s1 == 3 and R_s2 == 1
        and T_s0 == 3 and T_s1 == 1
    ):
        _rot_trans_kernel_contig[grid](
            out, x_centered, R, T, (mask_ptr if HAS_MASK else out),
            N,
            BLOCK=BLOCK,
            HAS_MASK=HAS_MASK,
            num_warps=4,
            num_stages=2,
        )
        return out

    # General path with arbitrary strides
    _rot_trans_kernel[grid](
        out, x_centered, R, T, mask_ptr,
        N,
        out_s0, out_s1, out_s2,
        x_s0, x_s1,
        R_s0, R_s1, R_s2,
        T_s0, T_s1,
        mask_s0,
        BLOCK=BLOCK,
        HAS_MASK=HAS_MASK,
        num_warps=4,
        num_stages=2,
    )
    return out


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    n_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Args:
        x_input_coords: [N_atom, 3]
        mask: [N_atom] 0/1 (可选)
    Returns:
        x_aug: [n_sample, N_atom, 3]
    """
    device = x_input_coords.device
    dtype = x_input_coords.dtype

    if mask is None:
        center = x_input_coords.mean(dim=-2, keepdim=True)
    else:
        m = mask.to(dtype=dtype).unsqueeze(-1)
        center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + eps)

    # Center the coordinates
    x0 = x_input_coords - center

    if centre_only:
        # Return the centered coordinates expanded to [n_sample, N, 3]
        return x0.unsqueeze(0).expand(n_sample, -1, -1).contiguous()

    # Generate random rotation and translation
    R = random_rotation_matrices(n_sample, device=device, dtype=dtype)  # [n,3,3]
    T = s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)

    # Apply rotation and translation using Triton kernel
    x = _apply_rot_trans_triton(x0, R, T, mask)

    # Reference applies mask only at the end; handled inside kernel already if provided.
    return x


class ModelNew(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(self, x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return centre_random_augmentation(
            x_input_coords=x_input_coords,
            n_sample=self.n_sample,
            s_trans=self.s_trans,
            centre_only=self.centre_only,
            mask=mask,
        )


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    device = 'cuda'
    torch.manual_seed(42)

    x_input_coords = torch.randn(N_ATOM, 3, device=device)
    mask = torch.ones(N_ATOM, device=device, dtype=torch.float32)

    return [x_input_coords, mask]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]