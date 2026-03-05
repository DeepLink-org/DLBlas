"""
Centre Random Augmentation (扩散采样里用于随机刚体变换)

From: protenix/model/utils.py:centre_random_augmentation
"""

import math
from typing import Optional
import torch
import torch.nn as nn

# Try to import Triton for kernel fusion
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def random_rotation_matrices(
    n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    生成 n 个随机旋转矩阵 [n,3,3]，基于随机四元数（均匀分布）。
    """
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)

    sqrt1 = torch.sqrt(1 - u1)  # sqrt(1-u1)
    sqrt_u = torch.sqrt(u1)  # sqrt(u1)
    two_pi_u2 = 2 * math.pi * u2
    two_pi_u3 = 2 * math.pi * u3
    sin_u2 = torch.sin(two_pi_u2)
    cos_u2 = torch.cos(two_pi_u2)
    sin_u3 = torch.sin(two_pi_u3)
    cos_u3 = torch.cos(two_pi_u3)

    x = sqrt1 * sin_u2
    y = sqrt1 * cos_u2
    z = sqrt_u * sin_u3
    w = sqrt_u * cos_u3

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


# Triton kernel for fused transformation (only defined if Triton is available)
if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=1),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def augmentation_kernel(
        x_ptr,
        center_ptr,
        R_ptr,
        T_ptr,
        mask_ptr,
        out_ptr,
        N,
        n_sample,
        stride_x0,
        stride_x1,
        stride_center0,
        stride_R0,
        stride_R1,
        stride_R2,
        stride_T0,
        stride_T1,
        stride_mask,
        stride_out0,
        stride_out1,
        stride_out2,
        has_mask: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        sample_idx = pid // num_chunks
        chunk_idx = pid % num_chunks

        # start index for this chunk
        start_idx = chunk_idx * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load atom coordinates (three components)
        x_vals = tl.load(
            x_ptr + offsets * stride_x0 + 0 * stride_x1, mask=mask, other=0.0
        )
        y_vals = tl.load(
            x_ptr + offsets * stride_x0 + 1 * stride_x1, mask=mask, other=0.0
        )
        z_vals = tl.load(
            x_ptr + offsets * stride_x0 + 2 * stride_x1, mask=mask, other=0.0
        )

        # Load center (scalars, broadcast later)
        center_x = tl.load(center_ptr + 0 * stride_center0)
        center_y = tl.load(center_ptr + 1 * stride_center0)
        center_z = tl.load(center_ptr + 2 * stride_center0)

        # Center the coordinates
        x_centered = x_vals - center_x
        y_centered = y_vals - center_y
        z_centered = z_vals - center_z

        # Load rotation matrix for this sample
        R_sample_ptr = R_ptr + sample_idx * stride_R0
        r00 = tl.load(R_sample_ptr + 0 * stride_R1 + 0 * stride_R2)
        r01 = tl.load(R_sample_ptr + 0 * stride_R1 + 1 * stride_R2)
        r02 = tl.load(R_sample_ptr + 0 * stride_R1 + 2 * stride_R2)
        r10 = tl.load(R_sample_ptr + 1 * stride_R1 + 0 * stride_R2)
        r11 = tl.load(R_sample_ptr + 1 * stride_R1 + 1 * stride_R2)
        r12 = tl.load(R_sample_ptr + 1 * stride_R1 + 2 * stride_R2)
        r20 = tl.load(R_sample_ptr + 2 * stride_R1 + 0 * stride_R2)
        r21 = tl.load(R_sample_ptr + 2 * stride_R1 + 1 * stride_R2)
        r22 = tl.load(R_sample_ptr + 2 * stride_R1 + 2 * stride_R2)

        # Load translation for this sample
        T_sample_ptr = T_ptr + sample_idx * stride_T0
        t0 = tl.load(T_sample_ptr + 0 * stride_T1)
        t1 = tl.load(T_sample_ptr + 1 * stride_T1)
        t2 = tl.load(T_sample_ptr + 2 * stride_T1)

        # Compute output vectors
        out0 = r00 * x_centered + r01 * y_centered + r02 * z_centered + t0
        out1 = r10 * x_centered + r11 * y_centered + r12 * z_centered + t1
        out2 = r20 * x_centered + r21 * y_centered + r22 * z_centered + t2

        # Apply mask if present
        if has_mask:
            mask_vals = tl.load(mask_ptr + offsets * stride_mask, mask=mask, other=0.0)
            out0 = out0 * mask_vals
            out1 = out1 * mask_vals
            out2 = out2 * mask_vals

        # Store results
        out_base = out_ptr + sample_idx * stride_out0 + offsets * stride_out1
        tl.store(out_base + 0 * stride_out2, out0, mask=mask)
        tl.store(out_base + 1 * stride_out2, out1, mask=mask)
        tl.store(out_base + 2 * stride_out2, out2, mask=mask)


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

    # Compute center (optionally masked)
    if mask is None:
        center = x_input_coords.mean(dim=-2, keepdim=True)
    else:
        m = mask.to(dtype=dtype).unsqueeze(-1)
        center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (
            m.sum(dim=-2, keepdim=True) + eps
        )

    if centre_only:
        # Expand to n_sample copies
        return (x_input_coords - center).unsqueeze(0).expand(n_sample, -1, -1)

    # Generate random rotations and translations
    R = random_rotation_matrices(n_sample, device=device, dtype=dtype)  # [n, 3, 3]
    T = s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)  # [n, 3]

    N = x_input_coords.size(0)

    # Use Triton kernel if possible
    use_triton = device.type == "cuda" and TRITON_AVAILABLE
    if use_triton:
        # Ensure contiguous tensors
        x = x_input_coords.contiguous()
        center_flat = center.reshape(-1).contiguous()
        R_contig = R.contiguous()
        T_contig = T.contiguous()
        out = torch.empty((n_sample, N, 3), device=device, dtype=dtype)

        # Strides (in elements)
        stride_x0, stride_x1 = x.stride()
        stride_center0 = center_flat.stride(0)
        stride_R0, stride_R1, stride_R2 = R_contig.stride()
        stride_T0, stride_T1 = T_contig.stride()
        stride_out0, stride_out1, stride_out2 = out.stride()

        if mask is not None:
            mask_contig = mask.to(dtype=dtype).contiguous()
            mask_ptr = mask_contig
            stride_mask = mask_contig.stride(0)
            has_mask = 1
        else:
            # dummy tensor (unused)
            mask_ptr = torch.empty(0, device=device, dtype=dtype)
            stride_mask = 0
            has_mask = 0

        # Grid lambda using meta parameters
        grid = lambda meta: (n_sample * triton.cdiv(N, meta["BLOCK_SIZE"]),)

        augmentation_kernel[grid](
            x,
            center_flat,
            R_contig,
            T_contig,
            mask_ptr,
            out,
            N,
            n_sample,
            stride_x0,
            stride_x1,
            stride_center0,
            stride_R0,
            stride_R1,
            stride_R2,
            stride_T0,
            stride_T1,
            stride_mask,
            stride_out0,
            stride_out1,
            stride_out2,
            has_mask=has_mask,
        )
        return out
    else:
        # Fallback to highly optimized PyTorch implementation
        x_centered = x_input_coords - center
        x_expanded = x_centered.unsqueeze(0).expand(n_sample, -1, -1)
        x_rot = torch.bmm(x_expanded, R.transpose(1, 2))
        out = x_rot + T.unsqueeze(1)
        if mask is not None:
            out = out * mask.to(dtype=dtype).view(1, -1, 1)
        return out


class ModelNew(nn.Module):
    def __init__(
        self, n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False
    ):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(
        self, x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
    device = "cuda"
    torch.manual_seed(42)

    x_input_coords = torch.randn(N_ATOM, 3, device=device)
    mask = torch.ones(N_ATOM, device=device, dtype=torch.float32)

    return [x_input_coords, mask]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
