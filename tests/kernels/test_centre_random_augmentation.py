"""
Centre Random Augmentation (扩散采样里用于随机刚体变换)

From: protenix/model/utils.py:centre_random_augmentation
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from dlblas.kernels.centre_random_augmentation import ModelNew


def random_rotation_matrices(
    n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
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
        center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (
            m.sum(dim=-2, keepdim=True) + eps
        )
    x = x_input_coords - center
    x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()

    if centre_only:
        return x

    R = random_rotation_matrices(n_sample, device=device, dtype=dtype)  # [n,3,3]
    T = s_trans * torch.randn(n_sample, 3, device=device, dtype=dtype)
    x = rot_vec_mul(R[:, None, :, :].expand(-1, x.shape[1], -1, -1), x) + T[:, None, :]

    if mask is not None:
        x = x * mask.to(dtype=dtype)[None, :, None]
    return x


class Model(nn.Module):
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


def test_centre_random_augmentation():
    x_input_coords, mask = get_inputs()
    model = Model()
    x = model.forward(x_input_coords, mask)

    x_input_coords, mask = get_inputs()
    model_new = ModelNew()
    x_new = model_new.forward(x_input_coords, mask)
    assert torch.allclose(x, x_new, rtol=1e-2, atol=1e-2)
