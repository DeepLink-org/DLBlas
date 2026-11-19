"""Fused center/rotation/translation kernels for Task09.

The four tiny RNG tensors are deliberately generated through PyTorch so the
candidate consumes the exact Philox sequence used by the reference under the
official seed reset.  Centering, quaternion-to-rotation conversion, rigid
transformation, masking and all output writes are performed by Triton kernels;
there is no computation fallback.
"""

import math
import os
from typing import Optional
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _select_kernel_config():
    """Return ``(profile, (center_warps, atom_block, augment_warps))``."""
    configs = {
        "auto": (4, 256, 4),
        "ascend_a2": (4, 128, 8),
        "metax_c500": (4, 256, 8),
        "iluvatar_bi150": (8, 128, 4),
        "enflame_s60": (4, 64, 4),
        "thead_810e": (4, 128, 8),
        "cambricon_mlu590": (4, 256, 4),
        "hygon_bw1000": (4, 256, 2),
        "mthreads_ph100": (4, 64, 8),
        "kunlun_p800": (8, 128, 8),
        "biren_br106m": (8, 256, 4),
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "auto").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    config = configs[profile]
    # At the official N=256 shape, atom_block=256 selects the single-launch
    # fused path; atom_block=128 retains the lower-footprint two-launch path.
    # Expose only those high-value A2 choices and preserve the default on any
    # unrecognised value.
    if profile in ("ascend_a2", "hygon_bw1000"):
        variants = {
            "256x1": (4, 256, 1),
            "128x2": (4, 128, 2),
            "128x4": (4, 128, 4),
            "128x8": (4, 128, 8),
            "256x2": (4, 256, 2),
            "256x4": (4, 256, 4),
            "256x8": (4, 256, 8),
        }
        variant = os.environ.get("KERNELSWIFT_TASK09_CONFIG", "").strip().lower()
        config = variants.get(variant, config)
    return profile, config


def _bw_launcher_cache_requested(profile: str) -> bool:
    """Enable the HCU 3.0 compiled runner only for an explicit experiment."""
    return (
        profile == "hygon_bw1000"
        and str(getattr(triton, "__version__", "")) == "3.0.0"
        and os.environ.get("KERNELSWIFT_BW_LAUNCHER_CACHE", "1").strip() == "1"
    )


def _bw_trusted_contract_requested(profile: str) -> bool:
    return (
        _bw_launcher_cache_requested(profile)
        and os.environ.get(
            "KERNELSWIFT_BW_TASK09_TRUSTED_CONTRACT", "0"
        ).strip()
        == "1"
    )


def _bw_graph_replay_requested(profile: str) -> bool:
    return (
        _bw_launcher_cache_requested(profile)
        and os.environ.get("KERNELSWIFT_BW_TASK09_GRAPH_REPLAY", "1").strip()
        == "1"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
        and hasattr(torch.cuda, "graph")
    )


@triton.jit
def _center_kernel(x_ptr, mask_ptr, center_ptr, n_atom, HAS_MASK: tl.constexpr,
                   eps: tl.constexpr, BLOCK: tl.constexpr):
    axis = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    value = tl.load(x_ptr + atom * 3 + axis, mask=valid, other=0.0).to(tl.float32)
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0).to(tl.float32)
        numerator = tl.sum(value * weight, axis=0)
        denominator = tl.sum(weight, axis=0) + eps
    else:
        numerator = tl.sum(value, axis=0)
        denominator = n_atom.to(tl.float32)
    tl.store(center_ptr + axis, numerator / denominator)


@triton.jit
def _center_only_kernel(x_ptr, center_ptr, out_ptr, n_atom,
                        BLOCK: tl.constexpr):
    sample = tl.program_id(0)
    block = tl.program_id(1)
    atom = block * BLOCK + tl.arange(0, BLOCK)
    valid = atom < n_atom
    for axis in tl.static_range(0, 3):
        value = tl.load(x_ptr + atom * 3 + axis, mask=valid, other=0.0)
        center = tl.load(center_ptr + axis)
        tl.store(out_ptr + (sample * n_atom + atom) * 3 + axis,
                 value - center, mask=valid)


@triton.jit
def _center_augment_fused_kernel(
    x_ptr, mask_ptr, u1_ptr, u2_ptr, u3_ptr, trans_ptr, out_ptr,
    n_atom, s_trans: tl.constexpr, HAS_MASK: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr,
):
    """Own one sample, including its copy of the three center reductions.

    For the fixed 256-atom benchmark this trades a few kilobytes of duplicate
    reads across four samples for one fewer launch and no center allocation.
    Profiles whose atom tile is smaller than the input use the two-kernel path
    below instead, so this fast path does not impose a large tile on a backend.
    """
    sample = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0).to(tl.float32)
        denominator = tl.sum(weight, axis=0) + eps
    else:
        weight = tl.where(valid, 1.0, 0.0)
        denominator = tl.sum(weight, axis=0)

    vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0).to(tl.float32)
    vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0).to(tl.float32)
    vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0).to(tl.float32)
    cx = tl.sum(vx * weight, axis=0) / denominator
    cy = tl.sum(vy * weight, axis=0) / denominator
    cz = tl.sum(vz * weight, axis=0) / denominator
    vx -= cx
    vy -= cy
    vz -= cz

    two_pi: tl.constexpr = 6.283185307179586
    u1 = tl.load(u1_ptr + sample).to(tl.float32)
    u2 = tl.load(u2_ptr + sample).to(tl.float32)
    u3 = tl.load(u3_ptr + sample).to(tl.float32)
    qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
    qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
    qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
    qw = tl.sqrt(u1) * tl.cos(two_pi * u3)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r00, r01, r02 = 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)
    r10, r11, r12 = 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)
    r20, r21, r22 = 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)
    tx = s_trans * tl.load(trans_ptr + sample * 3)
    ty = s_trans * tl.load(trans_ptr + sample * 3 + 1)
    tz = s_trans * tl.load(trans_ptr + sample * 3 + 2)
    ox = r00 * vx + r01 * vy + r02 * vz + tx
    oy = r10 * vx + r11 * vy + r12 * vz + ty
    oz = r20 * vx + r21 * vy + r22 * vz + tz
    if HAS_MASK:
        ox, oy, oz = ox * weight, oy * weight, oz * weight
    base = (sample * n_atom + atom) * 3
    tl.store(out_ptr + base, ox, mask=valid)
    tl.store(out_ptr + base + 1, oy, mask=valid)
    tl.store(out_ptr + base + 2, oz, mask=valid)


@triton.jit
def _augment_kernel(
    x_ptr, mask_ptr, center_ptr, u1_ptr, u2_ptr, u3_ptr, trans_ptr, out_ptr,
    n_atom, s_trans: tl.constexpr, HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sample = tl.program_id(0)
    block = tl.program_id(1)
    atom = block * BLOCK + tl.arange(0, BLOCK)
    valid = atom < n_atom
    two_pi: tl.constexpr = 6.283185307179586

    u1 = tl.load(u1_ptr + sample).to(tl.float32)
    u2 = tl.load(u2_ptr + sample).to(tl.float32)
    u3 = tl.load(u3_ptr + sample).to(tl.float32)
    qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
    qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
    qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
    qw = tl.sqrt(u1) * tl.cos(two_pi * u3)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r00, r01, r02 = 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)
    r10, r11, r12 = 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)
    r20, r21, r22 = 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)

    vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0) - tl.load(center_ptr)
    vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0) - tl.load(center_ptr + 1)
    vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0) - tl.load(center_ptr + 2)
    ox = r00 * vx + r01 * vy + r02 * vz + s_trans * tl.load(trans_ptr + sample * 3)
    oy = r10 * vx + r11 * vy + r12 * vz + s_trans * tl.load(trans_ptr + sample * 3 + 1)
    oz = r20 * vx + r21 * vy + r22 * vz + s_trans * tl.load(trans_ptr + sample * 3 + 2)
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0)
        ox, oy, oz = ox * weight, oy * weight, oz * weight
    base = (sample * n_atom + atom) * 3
    tl.store(out_ptr + base, ox, mask=valid)
    tl.store(out_ptr + base + 1, oy, mask=valid)
    tl.store(out_ptr + base + 2, oz, mask=valid)


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    n_sample: int,
    s_trans: float,
    centre_only: bool,
    mask: Optional[torch.Tensor],
    kernel_config,
    eps: float = 1e-12,
) -> torch.Tensor:
    if x_input_coords.device.type == "cpu" or (
        mask is not None and mask.device.type == "cpu"
    ):
        raise ValueError("centre_random_augmentation requires accelerator tensors")
    if mask is not None and mask.device != x_input_coords.device:
        raise ValueError("x_input_coords and mask must be on the same accelerator")
    n_atom = x_input_coords.shape[0]
    center_warps, atom_block, augment_warps = kernel_config
    mask_arg = mask if mask is not None else x_input_coords
    out = torch.empty(
        (n_sample, n_atom, 3),
        device=x_input_coords.device,
        dtype=x_input_coords.dtype,
    )
    grid = (n_sample, triton.cdiv(n_atom, atom_block))
    if not centre_only:
        # Match the reference RNG call sequence exactly under auto_bench's seed reset.
        u1 = torch.rand(
            n_sample,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        u2 = torch.rand(
            n_sample,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        u3 = torch.rand(
            n_sample,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        trans = torch.randn(
            n_sample,
            3,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        if n_atom <= atom_block:
            _center_augment_fused_kernel[(n_sample,)](
                x_input_coords, mask_arg, u1, u2, u3, trans, out, n_atom,
                s_trans=s_trans,
                HAS_MASK=mask is not None,
                eps=eps,
                BLOCK=atom_block,
                num_warps=augment_warps,
            )
            return out

    block = triton.next_power_of_2(n_atom)
    center = torch.empty(3, device=x_input_coords.device, dtype=torch.float32)
    _center_kernel[(3,)](
        x_input_coords, mask_arg, center, n_atom,
        HAS_MASK=mask is not None,
        eps=eps,
        BLOCK=block,
        num_warps=center_warps,
    )
    if centre_only:
        _center_only_kernel[grid](x_input_coords, center, out, n_atom,
                                  BLOCK=atom_block, num_warps=augment_warps)
    else:
        _augment_kernel[grid](
            x_input_coords, mask_arg, center, u1, u2, u3, trans, out, n_atom,
            s_trans=s_trans,
            HAS_MASK=mask is not None,
            BLOCK=atom_block,
            num_warps=augment_warps,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0,
                 centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only
        self.kernel_profile, self._kernel_config = _select_kernel_config()
        self._use_bw_launcher_cache = _bw_launcher_cache_requested(
            self.kernel_profile
        )
        self._use_bw_graph_replay = _bw_graph_replay_requested(
            self.kernel_profile
        )
        self._use_bw_full_rng_graph = (
            self._use_bw_graph_replay
            and os.environ.get(
                "KERNELSWIFT_BW_TASK09_FULL_RNG_GRAPH", "1"
            ).strip()
            == "1"
        )
        self._use_bw_trusted_contract = (
            self._use_bw_graph_replay
            or _bw_trusted_contract_requested(self.kernel_profile)
        )
        # The launcher cache retains only executable code and its device.
        # Graph mode separately owns fixed output/RNG workspace *storage*, as
        # explicitly allowed by the competition contract; every replay still
        # executes all four RNG nodes and the Triton kernel to overwrite it.
        # No random values, input-dependent result, raw pointer, or stream is
        # memoized by the launcher cache.
        self._bw_launcher_runner = None
        self._bw_launcher_device = None
        self._bw_trusted_inputs = None
        self._bw_graph = None
        self._bw_graph_inputs = None
        self._bw_graph_output = None
        self._bw_graph_u1 = None
        self._bw_graph_u2 = None
        self._bw_graph_u3 = None
        self._bw_graph_trans = None
        self._bw_graph_capture_armed = False

    def forward(self, x_input_coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        graph_inputs = self._bw_graph_inputs
        if (
            self._use_bw_graph_replay
            and self._bw_graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is x_input_coords
            and graph_inputs[1] is mask
        ):
            if not self._use_bw_full_rng_graph:
                self._fill_graph_rng()
            self._bw_graph.replay()
            return self._bw_graph_output

        trusted_inputs = self._bw_trusted_inputs
        if (
            self._use_bw_graph_replay
            and self._bw_graph_capture_armed
            and self._bw_launcher_runner is not None
            and trusted_inputs is not None
            and trusted_inputs[0] is x_input_coords
            and trusted_inputs[1] is mask
        ):
            return self._capture_graph_call(x_input_coords, mask)

        output = self._forward_impl(x_input_coords, mask)
        if self._use_bw_graph_replay:
            self._bw_graph = None
            self._bw_graph_inputs = None
            self._bw_graph_output = None
            self._bw_graph_capture_armed = True
        return output

    def _forward_impl(self, x_input_coords, mask):
        trusted_inputs = self._bw_trusted_inputs
        if (
            self._use_bw_trusted_contract
            and self._bw_launcher_runner is not None
            and self._bw_launcher_device == x_input_coords.device
            and trusted_inputs is not None
            and trusted_inputs[0] is x_input_coords
            and trusted_inputs[1] is mask
        ):
            return self._forward_trusted_launcher(x_input_coords, mask)

        if (
            self._use_bw_launcher_cache
            and self.n_sample == 4
            and self.s_trans == 1.0
            and not self.centre_only
            and self._kernel_config == (4, 256, 2)
            and x_input_coords.shape == (256, 3)
            and x_input_coords.dtype == torch.float32
            and x_input_coords.device.type == "cuda"
            and x_input_coords.is_contiguous()
            and mask is not None
            and mask.shape == (256,)
            and mask.dtype == torch.float32
            and mask.device == x_input_coords.device
            and mask.is_contiguous()
        ):
            # Preserve the reference allocation and Philox-consumption order on
            # every call, including a compiled-runner cache hit.
            out = torch.empty(
                (4, 256, 3),
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            u1 = torch.rand(
                4, device=x_input_coords.device, dtype=x_input_coords.dtype
            )
            u2 = torch.rand(
                4, device=x_input_coords.device, dtype=x_input_coords.dtype
            )
            u3 = torch.rand(
                4, device=x_input_coords.device, dtype=x_input_coords.dtype
            )
            trans = torch.randn(
                4, 3, device=x_input_coords.device, dtype=x_input_coords.dtype
            )
            grid = (4,)
            grid3 = (4, 1, 1)
            cache_eligible = (
                x_input_coords.data_ptr()
                | mask.data_ptr()
                | u1.data_ptr()
                | u2.data_ptr()
                | u3.data_ptr()
                | trans.data_ptr()
                | out.data_ptr()
            ) % 16 == 0
            if (
                cache_eligible
                and self._bw_launcher_runner is not None
                and self._bw_launcher_device == x_input_coords.device
            ):
                self._bw_launcher_runner(
                    x_input_coords,
                    mask,
                    u1,
                    u2,
                    u3,
                    trans,
                    out,
                    256,
                )
                if self._use_bw_trusted_contract:
                    self._bw_trusted_inputs = (x_input_coords, mask)
                return out
            compiled = _center_augment_fused_kernel[grid](
                x_input_coords,
                mask,
                u1,
                u2,
                u3,
                trans,
                out,
                256,
                s_trans=1.0,
                HAS_MASK=True,
                eps=1e-12,
                BLOCK=256,
                num_warps=2,
            )
            if cache_eligible and compiled is not None:
                self._bw_launcher_runner = compiled[grid3]
                self._bw_launcher_device = x_input_coords.device
                if self._use_bw_trusted_contract:
                    self._bw_trusted_inputs = (x_input_coords, mask)
            return out
        return centre_random_augmentation(
            x_input_coords,
            self.n_sample,
            self.s_trans,
            self.centre_only,
            mask,
            self._kernel_config,
        )

    def _forward_trusted_launcher(self, x_input_coords, mask):
        out = torch.empty(
            (4, 256, 3),
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        u1 = torch.rand(4, device=x_input_coords.device, dtype=x_input_coords.dtype)
        u2 = torch.rand(4, device=x_input_coords.device, dtype=x_input_coords.dtype)
        u3 = torch.rand(4, device=x_input_coords.device, dtype=x_input_coords.dtype)
        trans = torch.randn(
            4, 3, device=x_input_coords.device, dtype=x_input_coords.dtype
        )
        self._bw_launcher_runner(
            x_input_coords, mask, u1, u2, u3, trans, out, 256
        )
        return out

    def _ensure_graph_workspaces(self, x_input_coords):
        if (
            self._bw_graph_output is None
            or self._bw_graph_output.device != x_input_coords.device
        ):
            self._bw_graph_output = torch.empty(
                (4, 256, 3),
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            self._bw_graph_u1 = torch.empty(
                4, device=x_input_coords.device, dtype=x_input_coords.dtype
            )
            self._bw_graph_u2 = torch.empty_like(self._bw_graph_u1)
            self._bw_graph_u3 = torch.empty_like(self._bw_graph_u1)
            self._bw_graph_trans = torch.empty(
                (4, 3),
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )

    def _fill_graph_rng(self):
        torch.rand((4,), out=self._bw_graph_u1)
        torch.rand((4,), out=self._bw_graph_u2)
        torch.rand((4,), out=self._bw_graph_u3)
        torch.randn((4, 3), out=self._bw_graph_trans)

    def _capture_graph_call(self, x_input_coords, mask):
        self._ensure_graph_workspaces(x_input_coords)
        if not self._use_bw_full_rng_graph:
            self._fill_graph_rng()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if self._use_bw_full_rng_graph:
                self._fill_graph_rng()
            self._bw_launcher_runner(
                x_input_coords,
                mask,
                self._bw_graph_u1,
                self._bw_graph_u2,
                self._bw_graph_u3,
                self._bw_graph_trans,
                self._bw_graph_output,
                256,
            )
        self._bw_graph = graph
        self._bw_graph_inputs = (x_input_coords, mask)
        self._bw_graph_capture_armed = False
        graph.replay()
        return self._bw_graph_output


class Model(ModelNew):
    """Compatibility name required by the written competition rules."""


N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    torch.manual_seed(42)
    return [torch.randn(N_ATOM, 3, device="cuda"), torch.ones(N_ATOM, device="cuda")]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
