"""Fused center/rotation/translation kernels for Task09.

The default split path deliberately generates all four RNG tensors through the
same PyTorch calls as the reference, preserving its generator sequence under
the official seed reset.  The shared-storage candidate keeps the three uniform
draw calls but directs them into one fresh contiguous allocation.  Centering,
quaternion-to-rotation conversion, rigid transformation, masking and all output
writes are performed by Triton kernels; there is no computation fallback.
"""

import os
from importlib import metadata as importlib_metadata
from typing import Optional
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.compiler.compiler import CompiledKernel


def _is_triton_ascend_321_distribution():
    """Admit the private fixed-grid ABI only for audited Triton-Ascend 3.2.1.

    The 3.2.1 distribution reports ``triton.__version__ == "3.2.0"`` in the
    competition image.  Distribution metadata is therefore authoritative, and
    the ownership check prevents a different ``triton`` package from being
    accepted merely because the matching dist-info directory is also present.
    """
    distribution = next(
        (
            item
            for item in importlib_metadata.distributions()
            if item.metadata.get("Name", "").lower().replace("_", "-")
            == "triton-ascend"
        ),
        None,
    )
    if distribution is None:
        return False
    version = distribution.version.split("+", 1)[0]
    distribution_triton = os.path.realpath(
        str(distribution.locate_file("triton"))
    )
    imported_triton = os.path.realpath(str(triton.__file__))
    common_path = os.path.commonpath(
        (distribution_triton, imported_triton)
    )
    return (
        version == "3.2.1"
        and common_path == distribution_triton
        and callable(getattr(CompiledKernel, "__getitem__", None))
    )


def _compiled_launch_requested():
    """Return the Task09 launcher A/B choice; compiled is the default arm."""
    value = os.environ.get(
        "KERNELSWIFT_TASK09_COMPILED_LAUNCH", "compiled"
    ).strip().lower()
    if value in ("compiled", "1", "on", "true"):
        return True
    if value in ("jit", "0", "off", "false"):
        return False
    raise ValueError(
        "KERNELSWIFT_TASK09_COMPILED_LAUNCH must be compiled/jit or 1/0"
    )


def _select_rng_mode():
    """Select the reference-exact RNG call layout.

    ``split`` is deliberately the default and preserves the reference's three
    independent allocations and ``torch.rand(n_sample)`` calls.
    ``shared_storage`` preserves those three calls and their original order but
    directs them into row views of one fresh contiguous allocation.  The
    one-call packed experiment is intentionally not exposed because it does not
    preserve the 910B2 random stream beyond its first row.
    """
    value = os.environ.get(
        "KERNELSWIFT_TASK09_RNG_MODE", "split"
    ).strip().lower()
    if value not in ("split", "shared_storage"):
        raise ValueError(
            "KERNELSWIFT_TASK09_RNG_MODE must be split or shared_storage"
        )
    return value


def _compiled_grid_runner(compiled, grid3):
    """Extract a 3.2.1 grid-bound runner without retaining call-time tensors."""
    if not isinstance(compiled, CompiledKernel):
        return None
    runner = compiled[grid3]
    return runner if callable(runner) else None


def _matches_frozen_fused_abi(tensors):
    """Check every pointer specialization used by the official fused launch."""
    specs = (
        ((256, 3), (3, 1)),       # input coordinates
        ((256,), (1,)),           # mask
        ((4,), (1,)),             # u1
        ((4,), (1,)),             # u2
        ((4,), (1,)),             # u3
        ((4, 3), (3, 1)),         # translation
        ((4, 256, 3), (768, 3, 1)),  # output
    )
    if len(tensors) != len(specs):
        return False
    first_device = tensors[0].device
    if first_device.type != "npu":
        return False
    for tensor, (shape, stride) in zip(tensors, specs):
        if (
            tensor.device != first_device
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != stride
            or tensor.data_ptr() % 16 != 0
        ):
            return False
    return True


def _select_kernel_config():
    """Return ``(profile, (center_warps, atom_block, augment_warps))``."""
    configs = {
        "auto": (4, 256, 4),
        "ascend_a2": (4, 256, 2),
        "metax_c500": (4, 256, 8),
        "iluvatar_bi150": (8, 128, 4),
        "enflame_s60": (4, 64, 4),
        "thead_810e": (4, 128, 8),
        "cambricon_mlu590": (4, 256, 4),
        "hygon_bw1000": (8, 256, 8),
        "mthreads_ph100": (4, 64, 8),
        "kunlun_p800": (8, 128, 8),
        "biren_br106m": (8, 256, 4),
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "ascend_a2").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    config = configs[profile]
    # At the official N=256 shape, atom_block=256 selects the single-launch
    # fused path; atom_block=128 retains the lower-footprint two-launch path.
    # Expose only those high-value A2 choices and preserve the default on any
    # unrecognised value.
    if profile == "ascend_a2":
        variants = {
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


def _draw_augmentation_rng(n_sample, device, dtype, rng_mode):
    """Return fresh uniform row views followed by a fresh normal tensor.

    No tensor, storage, pointer, generator or stream escapes into launcher
    metadata.  A shared buffer stays alive through its row views for exactly the
    duration of the current forward call.
    """
    if rng_mode == "shared_storage":
        # Allocation and view creation do not touch the generator.  Keeping the
        # three rand launches separate and in reference order preserves both
        # their values and the successor RNG state while avoiding three storage
        # allocations.  The returned views keep ``uniforms`` alive naturally;
        # neither the storage nor a pointer is cached across forward calls.
        uniforms = torch.empty(
            (3, n_sample), device=device, dtype=dtype
        )
        u1, u2, u3 = uniforms.unbind(0)
        torch.rand(n_sample, device=device, dtype=dtype, out=u1)
        torch.rand(n_sample, device=device, dtype=dtype, out=u2)
        torch.rand(n_sample, device=device, dtype=dtype, out=u3)
    else:
        u1 = torch.rand(n_sample, device=device, dtype=dtype)
        u2 = torch.rand(n_sample, device=device, dtype=dtype)
        u3 = torch.rand(n_sample, device=device, dtype=dtype)
    trans = torch.randn(n_sample, 3, device=device, dtype=dtype)
    return u1, u2, u3, trans


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
    rng_mode="split",
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
    out = torch.empty((n_sample, n_atom, 3), device=x_input_coords.device,
                      dtype=x_input_coords.dtype)
    grid = (n_sample, triton.cdiv(n_atom, atom_block))
    if not centre_only:
        # ``split`` and ``shared_storage`` keep the reference's three RNG calls;
        # every mode performs fresh draws on every forward.
        u1, u2, u3, trans = _draw_augmentation_rng(
            n_sample,
            x_input_coords.device,
            x_input_coords.dtype,
            rng_mode,
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
        self._rng_mode = _select_rng_mode()
        self._use_compiled_launch = (
            _compiled_launch_requested()
            and _is_triton_ascend_321_distribution()
        )
        self._compiled_config_is_official = (
            self.kernel_profile == "ascend_a2"
            and self.n_sample == 4
            and self.s_trans == 1.0
            and not self.centre_only
            and self._kernel_config == (4, 256, 2)
        )
        # Retain executable code and immutable specialization metadata only.
        # Inputs, outputs, all four freshly generated RNG tensors, their
        # pointers, and the current NPU stream remain local to each forward.
        self._compiled_fused_launcher = None
        self._compiled_device = None
        self._compiled_abi_key = None
        self._compiled_kernel_metadata = None

    @staticmethod
    def _matches_official_inputs(x_input_coords, mask):
        """Guard before RNG generation so a fallback never advances RNG twice."""
        return (
            mask is not None
            and x_input_coords.device.type == "npu"
            and mask.device == x_input_coords.device
            and x_input_coords.dtype == torch.float32
            and mask.dtype == torch.float32
            and tuple(x_input_coords.shape) == (256, 3)
            and tuple(x_input_coords.stride()) == (3, 1)
            and tuple(mask.shape) == (256,)
            and tuple(mask.stride()) == (1,)
            and x_input_coords.data_ptr() % 16 == 0
            and mask.data_ptr() % 16 == 0
        )

    def _forward_frozen_fused(self, x_input_coords, mask):
        """Run and lazily publish the exact official 910B2 specialization."""
        # Keep fresh per-forward RNG draws ahead of the real kernel launch.
        # Split and shared_storage preserve the reference RNG sequence.  No
        # mode retains a random tensor between calls.
        out = torch.empty(
            (4, 256, 3),
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        u1, u2, u3, trans = _draw_augmentation_rng(
            4,
            x_input_coords.device,
            x_input_coords.dtype,
            self._rng_mode,
        )

        runtime_tensors = (
            x_input_coords, mask, u1, u2, u3, trans, out,
        )
        grid3 = (4, 1, 1)
        abi_key = (
            "triton-ascend-3.2.1",
            "task09-center-augment-fused",
            grid3,
            256,
            1.0,
            True,
            1e-12,
            256,
            2,
            "float32",
            (
                ((256, 3), (3, 1)),
                ((256,), (1,)),
                ((4,), (1,)),
                ((4,), (1,)),
                ((4,), (1,)),
                ((4, 3), (3, 1)),
                ((4, 256, 3), (768, 3, 1)),
            ),
            "aligned-16",
        )

        if (
            _matches_frozen_fused_abi(runtime_tensors)
            and self._compiled_fused_launcher is not None
            and self._compiled_device == x_input_coords.device
            and self._compiled_abi_key == abi_key
            and self._compiled_kernel_metadata is not None
        ):
            # Triton-Ascend 3.2.1's grid-bound runner takes non-constexpr ABI
            # arguments only.  It resolves the current NPU stream internally
            # for this real launch; no stream or call-time pointer is cached.
            self._compiled_fused_launcher(
                int(x_input_coords.data_ptr()),
                int(mask.data_ptr()),
                int(u1.data_ptr()),
                int(u2.data_ptr()),
                int(u3.data_ptr()),
                int(trans.data_ptr()),
                int(out.data_ptr()),
                256,
            )
            return out

        # The miss (including an alignment/layout miss after allocation) is a
        # normal JIT launch with exactly the already-generated RNG tensors.
        # Thus no fallback can repeat or perturb any of the four RNG calls.
        compiled = _center_augment_fused_kernel[grid3](
            x_input_coords, mask, u1, u2, u3, trans, out, 256,
            s_trans=1.0,
            HAS_MASK=True,
            eps=1e-12,
            BLOCK=256,
            num_warps=2,
        )
        if _matches_frozen_fused_abi(runtime_tensors):
            launcher = _compiled_grid_runner(compiled, grid3)
            compiled_metadata = getattr(compiled, "metadata", None)
            if launcher is not None and compiled_metadata is not None:
                self._compiled_fused_launcher = launcher
                self._compiled_device = x_input_coords.device
                self._compiled_kernel_metadata = compiled_metadata
                # Publish the immutable key last; it is the cache-valid marker
                # and deliberately contains no tensor identity or address.
                self._compiled_abi_key = abi_key
        return out

    def forward(self, x_input_coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (
            self._use_compiled_launch
            and self._compiled_config_is_official
            and self._matches_official_inputs(x_input_coords, mask)
        ):
            return self._forward_frozen_fused(x_input_coords, mask)
        return centre_random_augmentation(
            x_input_coords,
            self.n_sample,
            self.s_trans,
            self.centre_only,
            mask,
            self._kernel_config,
            self._rng_mode,
        )


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
