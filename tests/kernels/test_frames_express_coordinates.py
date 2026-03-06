"""
Frames: gather_frame_atom_by_indices + expressCoordinatesInFrame

From: protenix/model/modules/frames.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_frame_atom_by_indices(
    coordinate: torch.Tensor, frame_atom_index: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        coordinate: [N_atom, 3]
        frame_atom_index: [N_frame, 3] (每个 frame 由 3 个原子索引组成)
    Returns:
        frames: [N_frame, 3, 3]  (3 个原子 * 3 维坐标)
    """
    idx = frame_atom_index.long()
    x1 = coordinate.index_select(0, idx[:, 0])
    x2 = coordinate.index_select(0, idx[:, 1])
    x3 = coordinate.index_select(0, idx[:, 2])
    return torch.stack([x1, x2, x3], dim=1)


def expressCoordinatesInFrame(
    coordinate: torch.Tensor, frames: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Args:
        coordinate: [N_atom, 3]
        frames: [N_frame, 3, 3]  (a,b,c 三个点)
    Returns:
        x_transformed: [N_frame, N_atom, 3]
    """
    a, b, c = torch.unbind(frames, dim=-2)  # each: [N_frame, 3]
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)

    d = coordinate[None, :, :] - b[:, None, :]  # [N_frame, N_atom, 3]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[:, None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[:, None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[:, None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return x_transformed


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coordinate: torch.Tensor, frame_atom_index: torch.Tensor):
        frames = gather_frame_atom_by_indices(coordinate, frame_atom_index)
        return expressCoordinatesInFrame(coordinate, frames)


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_ATOM = 256
N_FRAME = 64


def get_inputs():
    device = "cuda"
    torch.manual_seed(42)

    coordinate = torch.randn(N_ATOM, 3, device=device)
    frame_atom_index = torch.randint(
        0, N_ATOM, (N_FRAME, 3), device=device, dtype=torch.int64
    )

    return [coordinate, frame_atom_index]


def get_init_inputs():
    return []
