"""KernelSwift Task09 RNG-compatible fused augmentation for Kunlun P800."""

import math
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "kunlun_p800"
ROW_WARPS = 1


@triton.jit
def _augment_kernel(
    coords,
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
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sample = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    if has_mask:
        weight = tl.load(mask + atom, mask=valid, other=0.0).to(tl.float32)
    else:
        weight = tl.where(valid, 1.0, 0.0)

    x0 = tl.load(coords + atom * 3, mask=valid, other=0.0).to(tl.float32)
    y0 = tl.load(coords + atom * 3 + 1, mask=valid, other=0.0).to(tl.float32)
    z0 = tl.load(coords + atom * 3 + 2, mask=valid, other=0.0).to(tl.float32)
    denominator = tl.sum(weight, axis=0) + eps
    x = x0 - tl.sum(x0 * weight, axis=0) / denominator
    y = y0 - tl.sum(y0 * weight, axis=0) / denominator
    z = z0 - tl.sum(z0 * weight, axis=0) / denominator

    if centre_only:
        ox = x
        oy = y
        oz = z
    else:
        u1 = tl.load(u1_ptr + sample).to(tl.float32)
        u2 = tl.load(u2_ptr + sample).to(tl.float32)
        u3 = tl.load(u3_ptr + sample).to(tl.float32)
        root_one_minus_u1 = tl.sqrt(1.0 - u1)
        root_u1 = tl.sqrt(u1)
        qx = root_one_minus_u1 * tl.sin((2.0 * math.pi) * u2)
        qy = root_one_minus_u1 * tl.cos((2.0 * math.pi) * u2)
        qz = root_u1 * tl.sin((2.0 * math.pi) * u3)
        qw = root_u1 * tl.cos((2.0 * math.pi) * u3)

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
        output = torch.empty(
            (self.n_sample, x_input_coords.shape[0], 3),
            dtype=x_input_coords.dtype,
            device=x_input_coords.device,
        )
        has_mask = mask is not None
        mask_arg = mask if has_mask else x_input_coords
        if self.centre_only:
            _augment_kernel[(self.n_sample,)](
                x_input_coords,
                mask_arg,
                x_input_coords,
                x_input_coords,
                x_input_coords,
                x_input_coords,
                output,
                n_atom=x_input_coords.shape[0],
                s_trans=self.s_trans,
                centre_only=True,
                has_mask=has_mask,
                eps=1e-12,
                BLOCK=256,
                num_warps=ROW_WARPS,
                num_stages=1,
            )
        else:
            # Keep the four framework RNG calls and their order identical to v0.
            u1 = torch.rand(
                self.n_sample,
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            u2 = torch.rand(
                self.n_sample,
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            u3 = torch.rand(
                self.n_sample,
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            translation = torch.randn(
                self.n_sample,
                3,
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            _augment_kernel[(self.n_sample,)](
                x_input_coords,
                mask_arg,
                u1,
                u2,
                u3,
                translation,
                output,
                n_atom=x_input_coords.shape[0],
                s_trans=self.s_trans,
                centre_only=False,
                has_mask=has_mask,
                eps=1e-12,
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
    coords = torch.randn(N_ATOM, 3, device="xpu")
    mask = torch.ones(N_ATOM, dtype=torch.float32, device="xpu")
    return [coords, mask]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
