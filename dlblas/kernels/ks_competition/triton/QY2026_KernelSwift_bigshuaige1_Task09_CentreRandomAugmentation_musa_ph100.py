"""KernelSwift Task09 fused augmentation candidate for PH100."""

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "musa_ph100"
ROW_WARPS = 4


@triton.jit
def _augment_centered_kernel(
    coords,
    center,
    mask,
    u1_ptr,
    u2_ptr,
    u3_ptr,
    translation,
    output,
    n_atom: tl.constexpr,
    s_trans: tl.constexpr,
    centre_only: tl.constexpr,
    has_mask: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sample = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    if has_mask:
        weight = tl.load(mask + atom, mask=valid, other=0.0).to(tl.float32)
    else:
        weight = tl.where(valid, 1.0, 0.0)

    cx = tl.load(center + 0).to(tl.float32)
    cy = tl.load(center + 1).to(tl.float32)
    cz = tl.load(center + 2).to(tl.float32)
    x = tl.load(coords + atom * 3, mask=valid, other=0.0).to(tl.float32) - cx
    y = tl.load(coords + atom * 3 + 1, mask=valid, other=0.0).to(tl.float32) - cy
    z = tl.load(coords + atom * 3 + 2, mask=valid, other=0.0).to(tl.float32) - cz

    if centre_only:
        ox, oy, oz = x, y, z
    else:
        q1 = tl.load(u1_ptr + sample).to(tl.float32)
        q2 = tl.load(u2_ptr + sample).to(tl.float32)
        q3 = tl.load(u3_ptr + sample).to(tl.float32)
        root_one_minus_q1 = tl.sqrt(1.0 - q1)
        root_q1 = tl.sqrt(q1)
        qx = root_one_minus_q1 * tl.sin((2.0 * math.pi) * q2)
        qy = root_one_minus_q1 * tl.cos((2.0 * math.pi) * q2)
        qz = root_q1 * tl.sin((2.0 * math.pi) * q3)
        qw = root_q1 * tl.cos((2.0 * math.pi) * q3)
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qw * qz)
        r02 = 2.0 * (qx * qz + qw * qy)
        r10 = 2.0 * (qx * qy + qw * qz)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qw * qx)
        r20 = 2.0 * (qx * qz - qw * qy)
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        tx = s_trans * tl.load(translation + sample * 3).to(tl.float32)
        ty = s_trans * tl.load(translation + sample * 3 + 1).to(tl.float32)
        tz = s_trans * tl.load(translation + sample * 3 + 2).to(tl.float32)
        ox = r00 * x + r01 * y + r02 * z + tx
        oy = r10 * x + r11 * y + r12 * z + ty
        oz = r20 * x + r21 * y + r22 * z + tz
        if has_mask:
            ox *= weight
            oy *= weight
            oz *= weight

    output_offset = (sample * n_atom + atom) * 3
    tl.store(output + output_offset, ox, mask=valid)
    tl.store(output + output_offset + 1, oy, mask=valid)
    tl.store(output + output_offset + 2, oz, mask=valid)


class ModelNew(nn.Module):
    def __init__(
        self,
        n_sample: int = 1,
        s_trans: float = 1.0,
        centre_only: bool = False,
    ):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only

    def forward(
        self,
        x_input_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = x_input_coords.dtype
        if mask is None:
            center = x_input_coords.mean(dim=-2, keepdim=True)
            mask_arg = x_input_coords
            has_mask = False
        else:
            m = mask.to(dtype=dtype).unsqueeze(-1)
            center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (
                m.sum(dim=-2, keepdim=True) + 1e-12
            )
            mask_arg = mask
            has_mask = True

        output = torch.empty(
            (self.n_sample, x_input_coords.shape[0], 3),
            dtype=dtype,
            device=x_input_coords.device,
        )
        if self.centre_only:
            u1 = u2 = u3 = translation = x_input_coords
        else:
            u1 = torch.rand(
                self.n_sample, device=x_input_coords.device, dtype=dtype
            )
            u2 = torch.rand(
                self.n_sample, device=x_input_coords.device, dtype=dtype
            )
            u3 = torch.rand(
                self.n_sample, device=x_input_coords.device, dtype=dtype
            )
            translation = torch.randn(
                self.n_sample, 3, device=x_input_coords.device, dtype=dtype
            )

        _augment_centered_kernel[(self.n_sample,)](
            x_input_coords,
            center,
            mask_arg,
            u1,
            u2,
            u3,
            translation,
            output,
            n_atom=x_input_coords.shape[0],
            s_trans=self.s_trans,
            centre_only=self.centre_only,
            has_mask=has_mask,
            BLOCK=256,
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        return output


class Model(ModelNew):
    pass


N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    torch.manual_seed(42)
    coords = torch.randn(N_ATOM, 3, device="cuda")
    mask = torch.ones(N_ATOM, dtype=torch.float32, device="cuda")
    return [coords, mask]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
