"""Triton implementation of Task04: SPLADESparsePooler."""

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from cuda.bindings import runtime as _cuda_runtime


# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, linear warps, linear stages,
#  layer-norm warps, pooling BLOCK_V, pooling warps)
_CHIP_CONFIGS = {
    "auto": (32, 64, 32, 8, 4, 3, 8, 64, 4),
    "ascend_a2": (32, 64, 32, 8, 4, 2, 8, 64, 4),
    # C500 has 64 threads/warp and 64 KiB shared memory per AP.  Stage 2 and
    # four layer-norm warps are the conservative first real-card baseline;
    # the alternatives below remain explicit A/B candidates.
    "metax_c500": (32, 64, 32, 8, 4, 1, 4, 64, 4),
    "iluvatar_bi150": (32, 128, 32, 8, 8, 3, 8, 64, 4),
    "enflame_s60": (32, 64, 32, 8, 4, 2, 4, 64, 4),
    "thead_810e": (16, 64, 32, 4, 4, 2, 4, 64, 4),
    "cambricon_mlu590": (16, 64, 32, 4, 4, 2, 4, 64, 4),
    "hygon_bw1000": (32, 128, 32, 8, 8, 2, 8, 128, 8),
    "mthreads_ph100": (32, 128, 32, 8, 8, 3, 8, 128, 8),
    "kunlun_p800": (16, 64, 32, 4, 4, 2, 4, 64, 4),
    # Three-round H200 sweep: split 64+32, N128/K64/W4/S3.  The first six
    # fields also tune the dense projection; every older geometry remains an
    # explicit KERNELSWIFT_TASK04_CONFIG candidate.
    "nvidia_h200": (32, 128, 64, 8, 4, 3, 8, 64, 4),
}

# (decoder BLOCK_N, BLOCK_K, warps, stages).  Kept separate so an A2 machine
# can A/B the large decoder without perturbing the first 768x768 projection.
_A2_DECODER_CONFIG = (32, 64, 4, 1)
_A2_CORE_COUNTS = {}
_H200_SM_COUNTS = {}

_C500_LAUNCH_VARIANTS = {
    # BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, linear warps, linear stages,
    # layer-norm warps, pooling BLOCK_V, pooling warps
    "m16n64k32_w2_s1": (16, 64, 32, 4, 2, 1, 2, 64, 2),
    "m32n64k32_w2_s1": (32, 64, 32, 8, 2, 1, 2, 64, 2),
    "m32n64k32_w4_s1": (32, 64, 32, 8, 4, 1, 4, 64, 4),
    "m32n64k32_w4_s2": (32, 64, 32, 8, 4, 2, 4, 64, 4),
    "m32n64k32_w4_s3": (32, 64, 32, 8, 4, 3, 8, 64, 4),
    "m32n128k32_w4_s2": (32, 128, 32, 8, 4, 2, 4, 64, 4),
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip() == "1"


def _get_h200_decoder_precision() -> str:
    """Accept only the decoder precision that passed the official H200 check."""
    key = os.environ.get(
        "KERNELSWIFT_TASK04_DECODER_PRECISION", "fp16"
    ).strip().lower()
    aliases = {
        "": "fp16",
        "default": "fp16",
        "safe": "fp16",
        "fp16": "fp16",
        "fp8": "fp8_row",
        "fp8_row": "fp8_row",
        "fp8_block128": "fp8_block128",
        "fp8_block64": "fp8_block64",
        "w8a16_row": "fp8_weight_row",
        "w8a16_block128": "fp8_weight_block128",
        "w8a16_block64": "fp8_weight_block64",
        "fp8_weight_row": "fp8_weight_row",
        "fp8_weight_block128": "fp8_weight_block128",
        "fp8_weight_block64": "fp8_weight_block64",
        "int8": "int8_block64",
        "int8_block64": "int8_block64",
    }
    selected = aliases.get(key)
    if selected is None:
        raise ValueError(f"unsupported Task04 decoder precision: {key}")
    if selected != "fp16":
        raise ValueError(
            "Task04 reduced-precision decoder is disabled: FP8 failed official "
            "H200 accuracy, while the fast INT8 W8A8 path exceeded tolerance "
            "on a multi-seed validation sweep"
        )
    return selected


def _get_h200_hopper_mode() -> str:
    """Select pointer/TMA and logical/persistent scheduling independently."""
    chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
    key = os.environ.get("KERNELSWIFT_TASK04_HOPPER_MODE", "pointer").strip().lower()
    aliases = {
        "": "pointer",
        "default": "pointer",
        "pointer": "pointer",
        "persistent": "persistent",
        "tma": "tma",
        "tma_ws": "tma_ws",
        "persistent_tma": "persistent_tma",
        "persistent_tma_ws": "persistent_tma_ws",
    }
    selected = aliases.get(key)
    if selected is None:
        raise ValueError(f"unsupported Task04 Hopper mode: {key}")
    if selected.endswith("_ws"):
        raise ValueError(
            "Task04 warp-specialized TMA modes are disabled: Triton 3.6 fails "
            "the NVGPUWarpSpecialization compilation pass for this kernel"
        )
    if chip != "nvidia_h200" and selected != "pointer":
        raise ValueError("Task04 Hopper scheduling candidates require nvidia_h200")
    return selected


def _task04_tma_allocator(size: int, alignment: int, stream):
    """Scratch allocator required by Triton 3.6 device tensor descriptors."""
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _h200_sm_count(device_index: int) -> int:
    """Return the physical SM count, with a bounded reproducible override."""
    override = os.environ.get("KERNELSWIFT_TASK04_NUM_SMS")
    cache_key = (device_index, override)
    if cache_key in _H200_SM_COUNTS:
        return _H200_SM_COUNTS[cache_key]
    if override is not None:
        count = int(override)
        if count < 1 or count > 1024:
            raise ValueError("Task04 NUM_SMS must be in [1, 1024]")
    else:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            device_index
        )
        count = int(properties["multiprocessor_count"])
    _H200_SM_COUNTS[cache_key] = count
    return count


def _get_h200_launch_config(default):
    """Parse the bounded H200 geometry sweep without changing its default.

    ``m32n{64,128,256}k{32,64}_w{4,8}_s{1,2,3}`` is intentionally the
    complete accepted grammar.  Every value is a power-of-two tile/standard
    CUDA launch setting already consumed by the generic Triton kernels below;
    rejecting typos prevents an A/B run from silently measuring the default.
    """
    key = os.environ.get("KERNELSWIFT_TASK04_CONFIG", "").strip().lower()
    if key in ("", "default"):
        return default
    fields = key.split("_")
    if (
        len(fields) != 3
        or not fields[0].startswith("m32n")
        or not fields[1].startswith("w")
        or not fields[2].startswith("s")
    ):
        raise ValueError(f"unsupported Task04 H200 config: {key}")
    matrix = fields[0][4:]
    matrix_fields = matrix.split("k")
    n_text = matrix_fields[0] if len(matrix_fields) == 2 else ""
    k_text = matrix_fields[1] if len(matrix_fields) == 2 else ""
    warp_text = fields[1][1:]
    stage_text = fields[2][1:]
    if not (
        n_text.isdigit()
        and k_text.isdigit()
        and warp_text.isdigit()
        and stage_text.isdigit()
    ):
        raise ValueError(f"unsupported Task04 H200 config: {key}")
    block_n = int(n_text)
    block_k = int(k_text)
    num_warps = int(warp_text)
    num_stages = int(stage_text)
    if (
        block_n not in (64, 128, 256)
        or block_k not in (32, 64)
        or num_warps not in (4, 8)
        or num_stages not in (1, 2, 3)
    ):
        raise ValueError(f"unsupported Task04 H200 config: {key}")
    return (32, block_n, block_k, 8, num_warps, num_stages, 8, 64, 4)


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
    default = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])
    if chip == "nvidia_h200":
        return _get_h200_launch_config(default)
    if chip != "metax_c500":
        return default
    variant = os.environ.get(
        "KERNELSWIFT_TASK04_CONFIG", "m32n64k32_w4_s1"
    ).strip().lower()
    if variant in ("", "default"):
        return default
    if variant not in _C500_LAUNCH_VARIANTS:
        raise ValueError(f"unsupported Task04 C500 config: {variant}")
    return _C500_LAUNCH_VARIANTS[variant]


def _get_a2_decoder_experiment():
    """Return an opt-in decoder path and its launch tuple.

    C500 defaults to the validated 64+32-row split implementation; the atomic
    path remains an explicit diagnostic variant.  Named variants make
    structural A/B tests reproducible without silently changing another
    chip's path.
    """
    chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
    mode = os.environ.get("KERNELSWIFT_TASK04_DECODER_MODE", "").strip().lower()
    variants = {
        # BLOCK_M, BLOCK_N, BLOCK_K, warps, stages
        "owner_32x32x32_w4_s1": (32, 32, 32, 4, 1),
        "owner_32x64x32_w4_s1": (32, 64, 32, 4, 1),
        "owner_32x64x32_w4_s2": (32, 64, 32, 4, 2),
        "owner_32x64x64_w4_s1": (32, 64, 64, 4, 1),
        "owner_32x64x32_w8_s1": (32, 64, 32, 8, 1),
        "owner_persistent_32x32x32_w4_s1": (32, 32, 32, 4, 1),
        "owner_persistent_32x64x32_w4_s1": (32, 64, 32, 4, 1),
        "partial_32x32x32_w4_s1_p256": (32, 32, 32, 4, 1),
        "partial_32x64x32_w4_s1_p256": (32, 64, 32, 4, 1),
        "partial_32x64x32_w4_s2_p256": (32, 64, 32, 4, 2),
        "partial_32x64x64_w4_s1_p256": (32, 64, 64, 4, 1),
        "partial_32x128x32_w4_s1_p256": (32, 128, 32, 4, 1),
        "split_64p32x32x32_w4_s1": (64, 32, 32, 4, 1),
        "split_64p32x32x32_w4_s2": (64, 32, 32, 4, 2),
        "split_64p32x64x32_w4_s1": (64, 64, 32, 4, 1),
        "split_64p32x64x32_w4_s2": (64, 64, 32, 4, 2),
        "split_64p32x64x64_w4_s1": (64, 64, 64, 4, 1),
        "split_64p32x64x32_w8_s1": (64, 64, 32, 8, 1),
        "raw_32x64x32_w4_s1_p32": (32, 64, 32, 4, 1),
        "raw_32x128x32_w4_s1_p32": (32, 128, 32, 4, 1),
        "full_128x16x32_w4_s1": (128, 16, 32, 4, 1),
        "full_128x32x32_w4_s1": (128, 32, 32, 4, 1),
        "full_128x32x64_w4_s1": (128, 32, 64, 4, 1),
        "full_128x32x32_w8_s1": (128, 32, 32, 8, 1),
    }
    if chip == "nvidia_h200":
        # The H200 sweep reuses only portable Triton decompositions.  Launch
        # geometry comes from the same bounded config as the atomic baseline,
        # so one environment tuple can compare algorithms at like-for-like
        # BLOCK_N/BLOCK_K/warp/stage settings.
        (
            _,
            block_n,
            block_k,
            _,
            num_warps,
            num_stages,
            _,
            _,
            _,
        ) = _get_launch_config()
        config = (32, block_n, block_k, num_warps, num_stages)
        if mode in ("", "default"):
            return "split_h200_64p32", (
                64, block_n, block_k, num_warps, num_stages
            )
        if mode == "atomic":
            return "atomic", config
        if mode in ("atomic_fp16", "atomic16"):
            return "atomic_fp16_h200", config
        aliases = {
            "owner": "owner_h200",
            "sequence": "sequence_h200",
            "sequence_owner": "sequence_h200",
            "sequential": "sequential_h200_64p32",
            "sequential64_32": "sequential_h200_64p32",
            "partial": "partial_h200",
            "raw": "raw_h200",
            "split": "split_h200_64p32",
            "split64_32": "split_h200_64p32",
            "split_64p32": "split_h200_64p32",
            "nphase64": "split_h200_nphase64",
            "split_nphase64": "split_h200_nphase64",
            "nphase32": "split_h200_nphase32",
            "split_nphase32": "split_h200_nphase32",
        }
        selected = aliases.get(mode)
        if selected is None:
            raise ValueError(f"unsupported Task04 H200 decoder mode: {mode}")
        if selected.startswith("split_"):
            config = (64, block_n, block_k, num_warps, num_stages)
        return selected, config
    if chip == "metax_c500":
        # These are portable Triton decompositions.  C500 deliberately excludes
        # the A2 persistent/full-row variants: their occupancy assumptions are
        # tied to Ascend core/UB geometry.  The 64+32 split path is the C500
        # default because it passed the official evaluator while the 4-warp
        # vector atomic path produced incomplete output on mcTriton 3.0.0.
        c500_modes = {
            "owner_32x32x32_w4_s1",
            "owner_32x64x32_w4_s1",
            "owner_32x64x32_w4_s2",
            "partial_32x32x32_w4_s1_p256",
            "partial_32x64x32_w4_s1_p256",
            "partial_32x64x32_w4_s2_p256",
            "raw_32x64x32_w4_s1_p32",
            "split_64p32x32x32_w4_s1",
            "split_64p32x32x32_w4_s2",
            "split_64p32x64x32_w4_s1",
            "split_64p32x64x32_w4_s2",
        }
        if mode in ("", "default"):
            mode = "split_64p32x64x32_w4_s1"
        if mode == "atomic":
            return "atomic", variants["owner_32x64x32_w4_s1"]
        if mode not in c500_modes:
            raise ValueError(f"unsupported Task04 C500 decoder mode: {mode}")
        return mode, variants[mode]
    if chip != "ascend_a2":
        return "atomic", variants["owner_32x64x32_w4_s1"]
    if mode == "" or mode == "default":
        # Final 910B2C geometry sweep selected BLOCK_N=64, BLOCK_K=64 and four
        # warps: stable200 measured 0.616664 ms, versus 0.699394 ms for the
        # otherwise identical BLOCK_K=32 path.  The 64+32 row decomposition
        # performs the same 96 padded rows as the atomic baseline while loading
        # each decoder weight tile once and directly storing all four maxima.
        mode = "split_64p32x64x64_w4_s1"
    if mode == "atomic" or mode not in variants:
        return "atomic", variants["owner_32x64x32_w4_s1"]
    return mode, variants[mode]


def _ascend_core_count(device_index: int) -> int:
    """Return the physical Cube-core count, with a reproducible override."""
    override = os.environ.get("KERNELSWIFT_TASK04_PROGRAMS")
    cache_key = (device_index, override)
    if cache_key in _A2_CORE_COUNTS:
        return _A2_CORE_COUNTS[cache_key]
    if override is not None:
        count = int(override)
        if count < 1 or count > 65535:
            raise ValueError("Task04 program count must be in [1, 65535]")
    else:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            device_index
        )
        count = int(
            properties.get(
                "num_aicore", properties.get("multiprocessor_count", 1)
            )
        )
    _A2_CORE_COUNTS[cache_key] = count
    return count


@triton.jit
def _linear_activation_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Tiled ``input @ weight.T + bias`` with an optional fused activation."""
    program = tl.program_id(0)
    programs_m = tl.cdiv(rows, BLOCK_M)
    programs_n = tl.cdiv(columns, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group = program // programs_per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(programs_m - first_m, GROUP_M)
    program_m = first_m + ((program % programs_per_group) % group_m)
    program_n = (program % programs_per_group) // group_m

    offs_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    input_tiles = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    weight_tiles = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        input_tile = tl.load(
            input_tiles,
            mask=(offs_m[:, None] < rows) & (start_k + offs_k[None, :] < reduction),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_tiles,
            mask=(offs_n[None, :] < columns) & (start_k + offs_k[:, None] < reduction),
            other=0.0,
        )
        # The A2 Cube path is substantially faster for fp16 operands while
        # retaining fp32 accumulation.  Inputs/parameters remain fp32 outside
        # the dot, and the competition tolerance comfortably covers the
        # sub-millipercent quantisation error on the official random case.
        if USE_FP16_DOT:
            input_tile = input_tile.to(tl.float16)
            weight_tile = weight_tile.to(tl.float16)
        accumulator += tl.dot(input_tile, weight_tile)
        input_tiles += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk

    accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < columns, other=0.0)[None, :]
    if ACTIVATION == 1:
        # PyTorch's exact GELU and this standard tanh approximation differ by
        # at most about 4.8e-4; keeping it fused avoids a full extra tensor pass.
        cubic = accumulator * accumulator * accumulator
        argument = 0.7978845608028654 * (accumulator + 0.044715 * cubic)
        tanh_argument = 2.0 / (1.0 + tl.exp(-2.0 * argument)) - 1.0
        accumulator = 0.5 * accumulator * (1.0 + tanh_argument)
    elif ACTIVATION == 2:
        # SPLADE activation: log1p(relu(x)).
        accumulator = tl.log(1.0 + tl.maximum(accumulator, 0.0))

    output_offsets = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        output_offsets,
        accumulator,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < columns),
    )


@triton.jit
def _h200_dense_gelu_layer_norm_kernel(
    input_ptr,
    dense_weight_ptr,
    dense_bias_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    output_ptr,
    normalized_ptr,
    quantized_ptr,
    quantized_scale_ptr,
    quantized_residual_ptr,
    quantized_residual_scale_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_nm: tl.constexpr,
    stride_nn: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rk: tl.constexpr,
    stride_rsm: tl.constexpr,
    stride_rsg: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    eps: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANTIZE_INT8: tl.constexpr,
    INT8_RESIDUAL_GROUPS: tl.constexpr,
):
    """Fuse the dense/GELU launch and LayerNorm launch on Hopper.

    Tensor Cores require an M tile of at least 16, but each CTA may own fewer
    real rows.  The candidate deliberately pads 4/8/16 rows to M16, streams N
    tiles through one accumulator, and uses the output as private global
    scratch for the two LayerNorm scans before normalizing it in place.  This
    removes one launch and one allocation without recomputing dense outputs.
    """
    row_lane = tl.arange(0, 16)
    row_start = tl.program_id(0) * ROWS_PER_PROGRAM
    offs_m = row_start + row_lane
    valid_rows = (row_lane < ROWS_PER_PROGRAM) & (offs_m < rows)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_sum = tl.zeros((16,), tl.float32)

    for start_n in tl.range(0, columns, BLOCK_N):
        column = start_n + offs_n
        accumulator = tl.zeros((16, BLOCK_N), tl.float32)
        input_tiles = (
            input_ptr
            + offs_m[:, None] * stride_im
            + offs_k[None, :] * stride_ik
        )
        weight_tiles = (
            dense_weight_ptr
            + column[None, :] * stride_wn
            + offs_k[:, None] * stride_wk
        )
        for start_k in range(0, reduction, BLOCK_K):
            valid_k = start_k + offs_k < reduction
            input_tile = tl.load(
                input_tiles,
                mask=valid_rows[:, None] & valid_k[None, :],
                other=0.0,
            ).to(tl.float16)
            weight_tile = tl.load(
                weight_tiles,
                mask=(column[None, :] < columns) & valid_k[:, None],
                other=0.0,
            ).to(tl.float16)
            accumulator += tl.dot(input_tile, weight_tile)
            input_tiles += BLOCK_K * stride_ik
            weight_tiles += BLOCK_K * stride_wk
        bias = tl.load(
            dense_bias_ptr + column,
            mask=column < columns,
            other=0.0,
        ).to(tl.float32)
        value = accumulator + bias[None, :]
        cubic = value * value * value
        argument = 0.7978845608028654 * (value + 0.044715 * cubic)
        tanh_argument = 2.0 / (1.0 + tl.exp(-2.0 * argument)) - 1.0
        value = 0.5 * value * (1.0 + tanh_argument)
        valid = valid_rows[:, None] & (column[None, :] < columns)
        value = tl.where(valid, value, 0.0)
        row_sum += tl.sum(value, axis=1)
        tl.store(
            output_ptr
            + offs_m[:, None] * stride_om
            + column[None, :] * stride_on,
            value,
            mask=valid,
        )

    # The same CTA owns every scratch element it reloads.  The barrier makes
    # the intended global store/load ordering explicit to Hopper and Triton.
    tl.debug_barrier()
    mean = row_sum / columns
    squared_sum = tl.zeros((16,), tl.float32)
    for start_n in range(0, columns, BLOCK_N):
        column = start_n + offs_n
        valid = valid_rows[:, None] & (column[None, :] < columns)
        value = tl.load(
            output_ptr
            + offs_m[:, None] * stride_om
            + column[None, :] * stride_on,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        centered = tl.where(valid, value - mean[:, None], 0.0)
        squared_sum += tl.sum(centered * centered, axis=1)
    rstd = tl.rsqrt(squared_sum / columns + eps)

    for start_n in range(0, columns, BLOCK_N):
        column = start_n + offs_n
        valid_columns = column < columns
        valid = valid_rows[:, None] & valid_columns[None, :]
        value = tl.load(
            output_ptr
            + offs_m[:, None] * stride_om
            + column[None, :] * stride_on,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            affine_weight_ptr + column,
            mask=valid_columns,
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            affine_bias_ptr + column,
            mask=valid_columns,
            other=0.0,
        ).to(tl.float32)
        result = (value - mean[:, None]) * rstd[:, None]
        result = result * scale[None, :] + shift[None, :]
        if QUANTIZE_INT8:
            # The decoder consumes one independently scaled 64-wide INT8 tile
            # at a time.  Quantize while the affine LayerNorm result is still
            # resident, avoiding a fourth scan and a separate launch.
            tl.static_assert(BLOCK_N % 64 == 0)
            groups_per_tile: tl.constexpr = BLOCK_N // 64
            grouped = tl.reshape(result, (16, groups_per_tile, 64))
            maximum = tl.max(tl.abs(grouped), axis=2)
            quant_scale = tl.maximum(maximum / 127.0, 1.0e-12)
            scaled = grouped / quant_scale[:, :, None]
            rounded = tl.where(
                scaled >= 0.0,
                tl.floor(scaled + 0.5),
                -tl.floor(-scaled + 0.5),
            )
            quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
            quantized = tl.reshape(quantized, (16, BLOCK_N))
            tl.store(
                quantized_ptr
                + offs_m[:, None] * stride_qm
                + column[None, :] * stride_qk,
                quantized,
                mask=valid,
            )
            group = start_n // 64 + tl.arange(0, groups_per_tile)
            tl.store(
                quantized_scale_ptr
                + offs_m[:, None] * stride_sm
                + group[None, :] * stride_sg,
                quant_scale,
                mask=valid_rows[:, None] & (group[None, :] < columns // 64),
            )
            if INT8_RESIDUAL_GROUPS > start_n // 64:
                residual = grouped - tl.reshape(
                    quantized, (16, groups_per_tile, 64)
                ) * quant_scale[:, :, None]
                residual_maximum = tl.max(tl.abs(residual), axis=2)
                residual_scale = tl.maximum(
                    residual_maximum / 127.0, 1.0e-12
                )
                residual_scaled = residual / residual_scale[:, :, None]
                residual_rounded = tl.where(
                    residual_scaled >= 0.0,
                    tl.floor(residual_scaled + 0.5),
                    -tl.floor(-residual_scaled + 0.5),
                )
                quantized_residual = tl.minimum(
                    tl.maximum(residual_rounded, -127.0), 127.0
                )
                quantized_residual = tl.reshape(
                    quantized_residual, (16, BLOCK_N)
                )
                tl.store(
                    quantized_residual_ptr
                    + offs_m[:, None] * stride_rm
                    + column[None, :] * stride_rk,
                    quantized_residual,
                    mask=valid,
                )
                tl.store(
                    quantized_residual_scale_ptr
                    + offs_m[:, None] * stride_rsm
                    + group[None, :] * stride_rsg,
                    residual_scale,
                    mask=valid_rows[:, None]
                    & (group[None, :] < columns // 64),
                )
        else:
            tl.store(
                normalized_ptr
                + offs_m[:, None] * stride_nm
                + column[None, :] * stride_nn,
                result,
                mask=valid,
            )


@triton.jit
def _zero_kernel(output_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output_ptr + offsets, 0.0, mask=offsets < total)


@triton.jit
def _decoder_max_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Fuse decoder GEMM, monotone SPLADE activation and per-segment max.

    Programs retain the cache-friendly global GEMM tiling.  Each tile reduces
    its rows for every intersecting sequence and atomically merges that partial
    maximum into the final [sequence, vocabulary] output.  Because
    log1p(relu(.)) is monotone, applying it after the tile max is exact.
    """
    program = tl.program_id(0)
    programs_m = tl.cdiv(rows, BLOCK_M)
    programs_n = tl.cdiv(columns, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group = program // programs_per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(programs_m - first_m, GROUP_M)
    program_m = first_m + ((program % programs_per_group) % group_m)
    program_n = (program % programs_per_group) // group_m

    offs_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    input_tiles = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    weight_tiles = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        input_tile = tl.load(
            input_tiles,
            mask=(offs_m[:, None] < rows) & (start_k + offs_k[None, :] < reduction),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_tiles,
            mask=(offs_n[None, :] < columns) & (start_k + offs_k[:, None] < reduction),
            other=0.0,
        )
        if USE_FP16_DOT:
            input_tile = input_tile.to(tl.float16)
            weight_tile = weight_tile.to(tl.float16)
        accumulator += tl.dot(input_tile, weight_tile)
        input_tiles += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    valid_vocab = offs_n < columns
    bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
    for sequence in range(NUM_SEQUENCES):
        start = tl.sum(tl.where(sequence_ids < sequence, lengths, 0), axis=0)
        length = tl.sum(tl.where(sequence_ids == sequence, lengths, 0), axis=0)
        row_mask = (offs_m >= start) & (offs_m < start + length) & (offs_m < rows)
        partial = tl.max(
            tl.where(row_mask[:, None], accumulator, -float("inf")), axis=0
        )
        activated = tl.log(1.0 + tl.maximum(partial + bias, 0.0))
        has_rows = tl.sum(row_mask.to(tl.int32), axis=0) > 0
        tl.atomic_max(
            pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
            activated,
            mask=valid_vocab & has_rows,
        )


@triton.jit
def _decoder_partial_max_kernel(
    input_ptr,
    weight_ptr,
    seq_lens_ptr,
    partial_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Decoder GEMM tile followed by non-atomic segment partial maxima."""
    program = tl.program_id(0)
    programs_m = tl.cdiv(rows, BLOCK_M)
    programs_n = tl.cdiv(columns, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group = program // programs_per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(programs_m - first_m, GROUP_M)
    program_m = first_m + ((program % programs_per_group) % group_m)
    program_n = (program % programs_per_group) // group_m

    offs_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    input_tiles = (
        input_ptr
        + offs_m[:, None] * stride_im
        + offs_k[None, :] * stride_ik
    )
    weight_tiles = (
        weight_ptr
        + offs_n[None, :] * stride_wn
        + offs_k[:, None] * stride_wk
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        input_tile = tl.load(
            input_tiles,
            mask=(offs_m[:, None] < rows)
            & (start_k + offs_k[None, :] < reduction),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_tiles,
            mask=(offs_n[None, :] < columns)
            & (start_k + offs_k[:, None] < reduction),
            other=0.0,
        )
        accumulator += tl.dot(input_tile, weight_tile)
        input_tiles += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk

    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    valid_vocab = offs_n < columns
    for sequence in range(NUM_SEQUENCES):
        sequence_start = tl.sum(
            tl.where(sequence_ids < sequence, lengths, 0), axis=0
        )
        sequence_length = tl.sum(
            tl.where(sequence_ids == sequence, lengths, 0), axis=0
        )
        row_mask = (
            (offs_m >= sequence_start)
            & (offs_m < sequence_start + sequence_length)
            & (offs_m < rows)
        )
        partial = tl.max(
            tl.where(row_mask[:, None], accumulator, -float("inf")), axis=0
        )
        tl.store(
            partial_ptr
            + (program_m * NUM_SEQUENCES + sequence) * columns
            + offs_n,
            partial,
            mask=valid_vocab,
        )


@triton.jit
def _decoder_finalize_partial_max_kernel(
    partial_ptr,
    bias_ptr,
    pooled_ptr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    columns: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    PROGRAMS_M: tl.constexpr,
    BLOCK_PROGRAMS_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Reduce row-tile partials, then apply decoder bias and SPLADE map."""
    sequence = tl.program_id(0)
    vocab_block = tl.program_id(1)
    row_tile = tl.arange(0, BLOCK_PROGRAMS_M)
    vocab = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
    valid_vocab = vocab < columns
    values = tl.load(
        partial_ptr
        + (row_tile[:, None] * NUM_SEQUENCES + sequence) * columns
        + vocab[None, :],
        mask=(row_tile[:, None] < PROGRAMS_M) & valid_vocab[None, :],
        other=-float("inf"),
    )
    maximum = tl.max(values, axis=0)
    bias = tl.load(bias_ptr + vocab, mask=valid_vocab, other=0.0)
    activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
    tl.store(
        pooled_ptr + sequence * stride_pb + vocab * stride_pv,
        activated,
        mask=valid_vocab,
    )


@triton.jit
def _decoder_vocab_owner_max_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Make one program the sole owner of a vocabulary tile.

    The portable kernel creates one program for every row/vocabulary tile and
    atomically merges six segment fragments in the official 83-row case.  This
    alternative serialises the three small row tiles *inside* one vocabulary
    owner.  It keeps only one ``BLOCK_M x BLOCK_N`` accumulator live, updates
    four tiny running maxima, then writes the final output directly.  Thus it
    preserves the modest UB footprint of the working kernel while removing the
    output-zeroing launch, all atomics, and two thirds of program scheduling.
    Repeated weight loads are identical in count to the atomic implementation
    and can be reused by the owner's cache between successive row tiles.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_vocab_blocks = tl.cdiv(columns, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)

    # With a logical grid this loop runs once.  With the persistent A2 grid it
    # distributes all vocabulary tiles cyclically over the physical compute cores,
    # matching Triton-Ascend's recommended CV-fusion scheduling pattern.
    for program_n in range(pid, num_vocab_blocks, num_programs):
        offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_vocab = offs_n < columns
        maxima = tl.full(
            (BLOCK_SEQUENCES, BLOCK_N), -float("inf"), tl.float32
        )

        for start_m in range(0, rows, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            input_tiles = (
                input_ptr
                + offs_m[:, None] * stride_im
                + offs_k[None, :] * stride_ik
            )
            weight_tiles = (
                weight_ptr
                + offs_n[None, :] * stride_wn
                + offs_k[:, None] * stride_wk
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for start_k in range(0, reduction, BLOCK_K):
                input_tile = tl.load(
                    input_tiles,
                    mask=(offs_m[:, None] < rows)
                    & (start_k + offs_k[None, :] < reduction),
                    other=0.0,
                )
                weight_tile = tl.load(
                    weight_tiles,
                    mask=valid_vocab[None, :]
                    & (start_k + offs_k[:, None] < reduction),
                    other=0.0,
                )
                accumulator += tl.dot(input_tile, weight_tile)
                input_tiles += BLOCK_K * stride_ik
                weight_tiles += BLOCK_K * stride_wk

            for sequence in range(NUM_SEQUENCES):
                sequence_start = tl.sum(
                    tl.where(sequence_ids < sequence, lengths, 0), axis=0
                )
                sequence_length = tl.sum(
                    tl.where(sequence_ids == sequence, lengths, 0), axis=0
                )
                row_mask = (
                    (offs_m >= sequence_start)
                    & (offs_m < sequence_start + sequence_length)
                    & (offs_m < rows)
                )
                partial = tl.max(
                    tl.where(row_mask[:, None], accumulator, -float("inf")),
                    axis=0,
                )
                owner = sequence_ids[:, None] == sequence
                maxima = tl.where(
                    owner, tl.maximum(maxima, partial[None, :]), maxima
                )

        bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
        activated = tl.log(1.0 + tl.maximum(maxima + bias[None, :], 0.0))
        tl.store(
            pooled_ptr
            + sequence_ids[:, None] * stride_pb
            + offs_n[None, :] * stride_pv,
            activated,
            mask=(sequence_ids[:, None] < NUM_SEQUENCES)
            & valid_vocab[None, :],
        )


@triton.jit
def _a2_decoder_split_rows_max_pool_kernel(
    input_ptr,
    input_scale_ptr,
    input_residual_ptr,
    input_residual_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
    USE_INT8_WEIGHT: tl.constexpr,
    INT8_RESIDUAL_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPECIALIZE_SEQ4: tl.constexpr,
):
    """Own all rows as supported 64+32 Cube tiles while loading W once.

    A monolithic padded M=128 tile wastes 54% of decoder work.  The official
    M=83 case fits in a 64-row tile plus a 32-row tile: both accumulators stay
    live across K, and each decoder weight tile feeds both dots before being
    discarded.  This retains the working power-of-two dot shapes, performs the
    same padded 96 rows as the atomic baseline, but removes repeated weight
    loads, output zeroing, and atomics.
    """
    program_n = tl.program_id(0)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m0 = tl.arange(0, 64)
    offs_m1 = 64 + tl.arange(0, 32)
    valid_vocab = offs_n < columns
    input_tiles0 = (
        input_ptr
        + offs_m0[:, None] * stride_im
        + offs_k[None, :] * stride_ik
    )
    input_tiles1 = (
        input_ptr
        + offs_m1[:, None] * stride_im
        + offs_k[None, :] * stride_ik
    )
    input_residual_tiles0 = (
        input_residual_ptr
        + offs_m0[:, None] * stride_im
        + offs_k[None, :] * stride_ik
    )
    input_residual_tiles1 = (
        input_residual_ptr
        + offs_m1[:, None] * stride_im
        + offs_k[None, :] * stride_ik
    )
    weight_tiles = (
        weight_ptr
        + offs_n[None, :] * stride_wn
        + offs_k[:, None] * stride_wk
    )
    accumulator0 = tl.zeros((64, BLOCK_N), tl.float32)
    accumulator1 = tl.zeros((32, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        input_tile0 = tl.load(
            input_tiles0,
            mask=(offs_m0[:, None] < rows)
            & (start_k + offs_k[None, :] < reduction),
            other=0.0,
        )
        input_tile1 = tl.load(
            input_tiles1,
            mask=(offs_m1[:, None] < rows)
            & (start_k + offs_k[None, :] < reduction),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_tiles,
            mask=valid_vocab[None, :]
            & (start_k + offs_k[:, None] < reduction),
            other=0.0,
        )
        if USE_INT8_WEIGHT:
            scale_group = start_k // 64
            input_scale0 = tl.load(
                input_scale_ptr + offs_m0 * 12 + scale_group,
                mask=offs_m0 < rows,
                other=0.0,
            ).to(tl.float32)
            input_scale1 = tl.load(
                input_scale_ptr + offs_m1 * 12 + scale_group,
                mask=offs_m1 < rows,
                other=0.0,
            ).to(tl.float32)
            weight_scale = tl.load(
                weight_scale_ptr + offs_n * 12 + scale_group,
                mask=valid_vocab,
                other=0.0,
            ).to(tl.float32)
            partial0 = tl.dot(
                input_tile0.to(tl.int8),
                weight_tile.to(tl.int8),
                out_dtype=tl.int32,
            ).to(tl.float32)
            partial1 = tl.dot(
                input_tile1.to(tl.int8),
                weight_tile.to(tl.int8),
                out_dtype=tl.int32,
            ).to(tl.float32)
            if INT8_RESIDUAL_GROUPS > scale_group:
                input_residual_tile0 = tl.load(
                    input_residual_tiles0,
                    mask=(offs_m0[:, None] < rows)
                    & (start_k + offs_k[None, :] < reduction),
                    other=0,
                )
                input_residual_tile1 = tl.load(
                    input_residual_tiles1,
                    mask=(offs_m1[:, None] < rows)
                    & (start_k + offs_k[None, :] < reduction),
                    other=0,
                )
                input_residual_scale0 = tl.load(
                    input_residual_scale_ptr + offs_m0 * 12 + scale_group,
                    mask=offs_m0 < rows,
                    other=0.0,
                ).to(tl.float32)
                input_residual_scale1 = tl.load(
                    input_residual_scale_ptr + offs_m1 * 12 + scale_group,
                    mask=offs_m1 < rows,
                    other=0.0,
                ).to(tl.float32)
                residual_partial0 = tl.dot(
                    input_residual_tile0.to(tl.int8),
                    weight_tile.to(tl.int8),
                    out_dtype=tl.int32,
                ).to(tl.float32)
                residual_partial1 = tl.dot(
                    input_residual_tile1.to(tl.int8),
                    weight_tile.to(tl.int8),
                    out_dtype=tl.int32,
                ).to(tl.float32)
                partial0 = (
                    partial0 * input_scale0[:, None]
                    + residual_partial0 * input_residual_scale0[:, None]
                )
                partial1 = (
                    partial1 * input_scale1[:, None]
                    + residual_partial1 * input_residual_scale1[:, None]
                )
            else:
                partial0 = partial0 * input_scale0[:, None]
                partial1 = partial1 * input_scale1[:, None]
            accumulator0 += partial0 * weight_scale[None, :]
            accumulator1 += partial1 * weight_scale[None, :]
        elif USE_FP16_DOT:
            input_tile0 = input_tile0.to(tl.float16)
            input_tile1 = input_tile1.to(tl.float16)
            weight_tile = weight_tile.to(tl.float16)
            accumulator0 += tl.dot(input_tile0, weight_tile)
            accumulator1 += tl.dot(input_tile1, weight_tile)
        else:
            accumulator0 += tl.dot(input_tile0, weight_tile)
            accumulator1 += tl.dot(input_tile1, weight_tile)
        input_tiles0 += BLOCK_K * stride_ik
        input_tiles1 += BLOCK_K * stride_ik
        input_residual_tiles0 += BLOCK_K * stride_ik
        input_residual_tiles1 += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk

    if SPECIALIZE_SEQ4:
        # Keep all four runtime lengths scalar.  This removes four vector
        # reductions per vocabulary tile while retaining the exact dynamic
        # segment boundaries supplied by seq_lens.
        length0 = tl.load(seq_lens_ptr).to(tl.int32)
        length1 = tl.load(seq_lens_ptr + 1).to(tl.int32)
        length2 = tl.load(seq_lens_ptr + 2).to(tl.int32)
        length3 = tl.load(seq_lens_ptr + 3).to(tl.int32)
    else:
        sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
        lengths = tl.load(
            seq_lens_ptr + sequence_ids,
            mask=sequence_ids < NUM_SEQUENCES,
            other=0,
        ).to(tl.int32)
    bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
    for sequence in range(NUM_SEQUENCES):
        if SPECIALIZE_SEQ4:
            if sequence == 0:
                sequence_start = 0
                sequence_length = length0
            elif sequence == 1:
                sequence_start = length0
                sequence_length = length1
            elif sequence == 2:
                sequence_start = length0 + length1
                sequence_length = length2
            else:
                sequence_start = length0 + length1 + length2
                sequence_length = length3
        else:
            sequence_start = tl.sum(
                tl.where(sequence_ids < sequence, lengths, 0), axis=0
            )
            sequence_length = tl.sum(
                tl.where(sequence_ids == sequence, lengths, 0), axis=0
            )
        row_mask0 = (
            (offs_m0 >= sequence_start)
            & (offs_m0 < sequence_start + sequence_length)
            & (offs_m0 < rows)
        )
        row_mask1 = (
            (offs_m1 >= sequence_start)
            & (offs_m1 < sequence_start + sequence_length)
            & (offs_m1 < rows)
        )
        maximum0 = tl.max(
            tl.where(row_mask0[:, None], accumulator0, -float("inf")),
            axis=0,
        )
        maximum1 = tl.max(
            tl.where(row_mask1[:, None], accumulator1, -float("inf")),
            axis=0,
        )
        maximum = tl.maximum(maximum0, maximum1)
        activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
        tl.store(
            pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
            activated,
            mask=valid_vocab,
        )


@triton.jit
def _h200_decoder_nphase_max_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    OUTER_N: tl.constexpr,
    INNER_N: tl.constexpr,
    PHASES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Own OUTER_N vocab columns but materialise one INNER_N phase at a time.

    The established N128 decoder keeps 96x128 FP32 accumulator elements live
    and reaches Hopper's 255-register ceiling.  This specialization preserves
    its 239-CTA logical grid and total decoder-weight traffic, while N64 (or
    N32) phases halve (or quarter) the live accumulator.  The 83x768 input is
    deliberately reread from L2 for each phase; it is tiny relative to the
    44.7 MiB decoder matrix and buys additional resident CTAs per SM.
    """
    program_n = tl.program_id(0)
    inner_lane = tl.arange(0, INNER_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m0 = tl.arange(0, 64)
    offs_m1 = 64 + tl.arange(0, 32)
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)

    for phase in range(PHASES):
        offs_n = program_n * OUTER_N + phase * INNER_N + inner_lane
        valid_vocab = offs_n < columns
        input_tiles0 = (
            input_ptr
            + offs_m0[:, None] * stride_im
            + offs_k[None, :] * stride_ik
        )
        input_tiles1 = (
            input_ptr
            + offs_m1[:, None] * stride_im
            + offs_k[None, :] * stride_ik
        )
        weight_tiles = (
            weight_ptr
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk
        )
        accumulator0 = tl.zeros((64, INNER_N), tl.float32)
        accumulator1 = tl.zeros((32, INNER_N), tl.float32)
        for start_k in range(0, reduction, BLOCK_K):
            input_tile0 = tl.load(
                input_tiles0,
                mask=(offs_m0[:, None] < rows)
                & (start_k + offs_k[None, :] < reduction),
                other=0.0,
            ).to(tl.float16)
            input_tile1 = tl.load(
                input_tiles1,
                mask=(offs_m1[:, None] < rows)
                & (start_k + offs_k[None, :] < reduction),
                other=0.0,
            ).to(tl.float16)
            weight_tile = tl.load(
                weight_tiles,
                mask=valid_vocab[None, :]
                & (start_k + offs_k[:, None] < reduction),
                other=0.0,
            ).to(tl.float16)
            accumulator0 += tl.dot(input_tile0, weight_tile)
            accumulator1 += tl.dot(input_tile1, weight_tile)
            input_tiles0 += BLOCK_K * stride_ik
            input_tiles1 += BLOCK_K * stride_ik
            weight_tiles += BLOCK_K * stride_wk

        bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
        for sequence in range(NUM_SEQUENCES):
            sequence_start = tl.sum(
                tl.where(sequence_ids < sequence, lengths, 0), axis=0
            )
            sequence_length = tl.sum(
                tl.where(sequence_ids == sequence, lengths, 0), axis=0
            )
            row_mask0 = (
                (offs_m0 >= sequence_start)
                & (offs_m0 < sequence_start + sequence_length)
                & (offs_m0 < rows)
            )
            row_mask1 = (
                (offs_m1 >= sequence_start)
                & (offs_m1 < sequence_start + sequence_length)
                & (offs_m1 < rows)
            )
            maximum0 = tl.max(
                tl.where(row_mask0[:, None], accumulator0, -float("inf")),
                axis=0,
            )
            maximum1 = tl.max(
                tl.where(row_mask1[:, None], accumulator1, -float("inf")),
                axis=0,
            )
            maximum = tl.maximum(maximum0, maximum1)
            activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
            tl.store(
                pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
                activated,
                mask=valid_vocab,
            )


@triton.jit
def _h200_decoder_sequence_owner_max_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Assign one sequence/vocabulary tile to each Hopper CTA.

    The official four sequences are at most 25 rows, so BLOCK_M=32 performs
    128 padded rows versus 96 for the established 64+32 owner.  In exchange,
    the accumulator footprint is one third as large and the grid exposes four
    times as many independent CTAs.  H200's 60 MiB L2 can retain the 44.7 MiB
    FP16 decoder weight, making the deliberate cross-sequence weight reload a
    plausible occupancy-for-cache trade.
    """
    sequence = tl.program_id(0)
    program_n = tl.program_id(1)
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    sequence_start = tl.sum(
        tl.where(sequence_ids < sequence, lengths, 0), axis=0
    )
    sequence_length = tl.sum(
        tl.where(sequence_ids == sequence, lengths, 0), axis=0
    )

    offs_m = sequence_start + tl.arange(0, BLOCK_M)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    valid_rows = (
        (tl.arange(0, BLOCK_M) < sequence_length) & (offs_m < rows)
    )
    valid_vocab = offs_n < columns
    input_tiles = (
        input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    weight_tiles = (
        weight_ptr
        + offs_n[None, :] * stride_wn
        + offs_k[:, None] * stride_wk
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        valid_k = start_k + offs_k < reduction
        input_tile = tl.load(
            input_tiles,
            mask=valid_rows[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float16)
        weight_tile = tl.load(
            weight_tiles,
            mask=valid_vocab[None, :] & valid_k[:, None],
            other=0.0,
        ).to(tl.float16)
        accumulator += tl.dot(input_tile, weight_tile)
        input_tiles += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk

    maximum = tl.max(
        tl.where(valid_rows[:, None], accumulator, -float("inf")), axis=0
    )
    bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
    activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
    tl.store(
        pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
        activated,
        mask=valid_vocab,
    )


@triton.jit
def _h200_decoder_sequential_64p32_max_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Process the 64- and 32-row decoder tiles sequentially.

    This preserves the established 96 padded rows, one-launch direct stores,
    and exact pooling.  It reloads the FP16 weight tile for the second phase so
    only one large accumulator is live at a time, trading H200 L2 bandwidth
    for substantially lower register/shared-memory pressure than the 255-reg
    parallel-accumulator default.
    """
    program_n = tl.program_id(0)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m0 = tl.arange(0, 64)
    offs_m1 = 64 + tl.arange(0, 32)
    valid_vocab = offs_n < columns
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    maxima = tl.full(
        (BLOCK_SEQUENCES, BLOCK_N), -float("inf"), tl.float32
    )

    input_tiles0 = (
        input_ptr + offs_m0[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    weight_tiles0 = (
        weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )
    accumulator0 = tl.zeros((64, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        valid_k = start_k + offs_k < reduction
        input_tile0 = tl.load(
            input_tiles0,
            mask=(offs_m0[:, None] < rows) & valid_k[None, :],
            other=0.0,
        ).to(tl.float16)
        weight_tile0 = tl.load(
            weight_tiles0,
            mask=valid_vocab[None, :] & valid_k[:, None],
            other=0.0,
        ).to(tl.float16)
        accumulator0 += tl.dot(input_tile0, weight_tile0)
        input_tiles0 += BLOCK_K * stride_ik
        weight_tiles0 += BLOCK_K * stride_wk
    for sequence in range(NUM_SEQUENCES):
        sequence_start = tl.sum(
            tl.where(sequence_ids < sequence, lengths, 0), axis=0
        )
        sequence_length = tl.sum(
            tl.where(sequence_ids == sequence, lengths, 0), axis=0
        )
        row_mask0 = (
            (offs_m0 >= sequence_start)
            & (offs_m0 < sequence_start + sequence_length)
            & (offs_m0 < rows)
        )
        partial0 = tl.max(
            tl.where(row_mask0[:, None], accumulator0, -float("inf")), axis=0
        )
        owner = sequence_ids[:, None] == sequence
        maxima = tl.where(owner, tl.maximum(maxima, partial0[None, :]), maxima)

    input_tiles1 = (
        input_ptr + offs_m1[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    weight_tiles1 = (
        weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )
    accumulator1 = tl.zeros((32, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        valid_k = start_k + offs_k < reduction
        input_tile1 = tl.load(
            input_tiles1,
            mask=(offs_m1[:, None] < rows) & valid_k[None, :],
            other=0.0,
        ).to(tl.float16)
        weight_tile1 = tl.load(
            weight_tiles1,
            mask=valid_vocab[None, :] & valid_k[:, None],
            other=0.0,
        ).to(tl.float16)
        accumulator1 += tl.dot(input_tile1, weight_tile1)
        input_tiles1 += BLOCK_K * stride_ik
        weight_tiles1 += BLOCK_K * stride_wk
    for sequence in range(NUM_SEQUENCES):
        sequence_start = tl.sum(
            tl.where(sequence_ids < sequence, lengths, 0), axis=0
        )
        sequence_length = tl.sum(
            tl.where(sequence_ids == sequence, lengths, 0), axis=0
        )
        row_mask1 = (
            (offs_m1 >= sequence_start)
            & (offs_m1 < sequence_start + sequence_length)
            & (offs_m1 < rows)
        )
        partial1 = tl.max(
            tl.where(row_mask1[:, None], accumulator1, -float("inf")), axis=0
        )
        owner = sequence_ids[:, None] == sequence
        maxima = tl.where(owner, tl.maximum(maxima, partial1[None, :]), maxima)

    bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
    activated = tl.log(1.0 + tl.maximum(maxima + bias[None, :], 0.0))
    tl.store(
        pooled_ptr
        + sequence_ids[:, None] * stride_pb
        + offs_n[None, :] * stride_pv,
        activated,
        mask=(sequence_ids[:, None] < NUM_SEQUENCES) & valid_vocab[None, :],
    )


@triton.jit
def _h200_decoder_ln_stats_max_pool_kernel(
    input_ptr,
    stats_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    decoder_weight_ptr,
    decoder_bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuse LayerNorm materialization into the split decoder.

    A tiny preceding kernel stores only mean/rstd for each row.  Vocabulary
    owners then apply the affine normalization while loading each decoder K
    tile.  Compared with the default this removes the full 83x768 normalized
    tensor write and lets dense output plus statistics share one allocation.
    Decoder arithmetic and FP16 Tensor Core operands remain unchanged.
    """
    program_n = tl.program_id(0)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m0 = tl.arange(0, 64)
    offs_m1 = 64 + tl.arange(0, 32)
    valid_vocab = offs_n < columns
    valid_m0 = offs_m0 < rows
    valid_m1 = offs_m1 < rows

    mean0 = tl.load(
        stats_ptr + offs_m0 * stride_sm,
        mask=valid_m0,
        other=0.0,
    ).to(tl.float32)
    rstd0 = tl.load(
        stats_ptr + offs_m0 * stride_sm + stride_ss,
        mask=valid_m0,
        other=0.0,
    ).to(tl.float32)
    mean1 = tl.load(
        stats_ptr + offs_m1 * stride_sm,
        mask=valid_m1,
        other=0.0,
    ).to(tl.float32)
    rstd1 = tl.load(
        stats_ptr + offs_m1 * stride_sm + stride_ss,
        mask=valid_m1,
        other=0.0,
    ).to(tl.float32)

    input_tiles0 = (
        input_ptr + offs_m0[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    input_tiles1 = (
        input_ptr + offs_m1[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    weight_tiles = (
        decoder_weight_ptr
        + offs_n[None, :] * stride_wn
        + offs_k[:, None] * stride_wk
    )
    accumulator0 = tl.zeros((64, BLOCK_N), tl.float32)
    accumulator1 = tl.zeros((32, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        valid_k = start_k + offs_k < reduction
        input_tile0 = tl.load(
            input_tiles0,
            mask=valid_m0[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)
        input_tile1 = tl.load(
            input_tiles1,
            mask=valid_m1[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)
        affine_scale = tl.load(
            affine_weight_ptr + start_k + offs_k,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        affine_shift = tl.load(
            affine_bias_ptr + start_k + offs_k,
            mask=valid_k,
            other=0.0,
        ).to(tl.float32)
        input_tile0 = (
            (input_tile0 - mean0[:, None])
            * rstd0[:, None]
            * affine_scale[None, :]
            + affine_shift[None, :]
        ).to(tl.float16)
        input_tile1 = (
            (input_tile1 - mean1[:, None])
            * rstd1[:, None]
            * affine_scale[None, :]
            + affine_shift[None, :]
        ).to(tl.float16)
        weight_tile = tl.load(
            weight_tiles,
            mask=valid_vocab[None, :] & valid_k[:, None],
            other=0.0,
        ).to(tl.float16)
        accumulator0 += tl.dot(input_tile0, weight_tile)
        accumulator1 += tl.dot(input_tile1, weight_tile)
        input_tiles0 += BLOCK_K * stride_ik
        input_tiles1 += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk

    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    bias = tl.load(decoder_bias_ptr + offs_n, mask=valid_vocab, other=0.0)
    for sequence in range(NUM_SEQUENCES):
        sequence_start = tl.sum(
            tl.where(sequence_ids < sequence, lengths, 0), axis=0
        )
        sequence_length = tl.sum(
            tl.where(sequence_ids == sequence, lengths, 0), axis=0
        )
        row_mask0 = (
            (offs_m0 >= sequence_start)
            & (offs_m0 < sequence_start + sequence_length)
            & valid_m0
        )
        row_mask1 = (
            (offs_m1 >= sequence_start)
            & (offs_m1 < sequence_start + sequence_length)
            & valid_m1
        )
        maximum0 = tl.max(
            tl.where(row_mask0[:, None], accumulator0, -float("inf")), axis=0
        )
        maximum1 = tl.max(
            tl.where(row_mask1[:, None], accumulator1, -float("inf")), axis=0
        )
        maximum = tl.maximum(maximum0, maximum1)
        activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
        tl.store(
            pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
            activated,
            mask=valid_vocab,
        )


@triton.jit
def _h200_decoder_split_rows_persistent_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_ism: tl.constexpr,
    stride_isk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wsm: tl.constexpr,
    stride_wsk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_WEIGHT_ONLY: tl.constexpr,
    FP8_SCALE_BLOCK_K: tl.constexpr,
    PROGRAM_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """H200 64+32 decoder with optional FP8 and a physical-SM grid.

    A logical grid still executes one vocabulary block per program.  A grid
    capped at NUM_SMS walks the vocabulary cyclically, providing a genuine
    persistent candidate without changing the numerical decomposition.
    """
    pid = tl.program_id(0)
    num_vocab_blocks = tl.cdiv(columns, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m0 = tl.arange(0, 64)
    offs_m1 = 64 + tl.arange(0, 32)
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)

    for program_n in tl.range(
        pid, num_vocab_blocks, PROGRAM_STEP, flatten=True
    ):
        offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_vocab = offs_n < columns
        input_tiles0 = (
            input_ptr
            + offs_m0[:, None] * stride_im
            + offs_k[None, :] * stride_ik
        )
        input_tiles1 = (
            input_ptr
            + offs_m1[:, None] * stride_im
            + offs_k[None, :] * stride_ik
        )
        weight_tiles = (
            weight_ptr
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk
        )
        accumulator0 = tl.zeros((64, BLOCK_N), tl.float32)
        accumulator1 = tl.zeros((32, BLOCK_N), tl.float32)
        for start_k in range(0, reduction, BLOCK_K):
            input_tile0 = tl.load(
                input_tiles0,
                mask=(offs_m0[:, None] < rows)
                & (start_k + offs_k[None, :] < reduction),
                other=0.0,
            )
            input_tile1 = tl.load(
                input_tiles1,
                mask=(offs_m1[:, None] < rows)
                & (start_k + offs_k[None, :] < reduction),
                other=0.0,
            )
            weight_tile = tl.load(
                weight_tiles,
                mask=valid_vocab[None, :]
                & (start_k + offs_k[:, None] < reduction),
                other=0.0,
            )
            if USE_FP8:
                scale_group = start_k // FP8_SCALE_BLOCK_K
                input_scale0 = tl.load(
                    input_scale_ptr
                    + offs_m0 * stride_ism
                    + scale_group * stride_isk,
                    mask=offs_m0 < rows,
                    other=0.0,
                ).to(tl.float32)
                input_scale1 = tl.load(
                    input_scale_ptr
                    + offs_m1 * stride_ism
                    + scale_group * stride_isk,
                    mask=offs_m1 < rows,
                    other=0.0,
                ).to(tl.float32)
                weight_scale = tl.load(
                    weight_scale_ptr
                    + offs_n * stride_wsm
                    + scale_group * stride_wsk,
                    mask=valid_vocab,
                    other=0.0,
                ).to(tl.float32)
                partial0 = tl.dot(input_tile0, weight_tile)
                partial1 = tl.dot(input_tile1, weight_tile)
                accumulator0 += (
                    partial0 * input_scale0[:, None] * weight_scale[None, :]
                )
                accumulator1 += (
                    partial1 * input_scale1[:, None] * weight_scale[None, :]
                )
            elif FP8_WEIGHT_ONLY:
                scale_group = start_k // FP8_SCALE_BLOCK_K
                weight_scale = tl.load(
                    weight_scale_ptr
                    + offs_n * stride_wsm
                    + scale_group * stride_wsk,
                    mask=valid_vocab,
                    other=0.0,
                ).to(tl.float32)
                dequantized_weight = (
                    weight_tile.to(tl.float32) * weight_scale[None, :]
                ).to(tl.float16)
                accumulator0 += tl.dot(
                    input_tile0.to(tl.float16), dequantized_weight
                )
                accumulator1 += tl.dot(
                    input_tile1.to(tl.float16), dequantized_weight
                )
            else:
                accumulator0 += tl.dot(
                    input_tile0.to(tl.float16), weight_tile.to(tl.float16)
                )
                accumulator1 += tl.dot(
                    input_tile1.to(tl.float16), weight_tile.to(tl.float16)
                )
            input_tiles0 += BLOCK_K * stride_ik
            input_tiles1 += BLOCK_K * stride_ik
            weight_tiles += BLOCK_K * stride_wk

        bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
        for sequence in range(NUM_SEQUENCES):
            sequence_start = tl.sum(
                tl.where(sequence_ids < sequence, lengths, 0), axis=0
            )
            sequence_length = tl.sum(
                tl.where(sequence_ids == sequence, lengths, 0), axis=0
            )
            row_mask0 = (
                (offs_m0 >= sequence_start)
                & (offs_m0 < sequence_start + sequence_length)
                & (offs_m0 < rows)
            )
            row_mask1 = (
                (offs_m1 >= sequence_start)
                & (offs_m1 < sequence_start + sequence_length)
                & (offs_m1 < rows)
            )
            maximum0 = tl.max(
                tl.where(row_mask0[:, None], accumulator0, -float("inf")),
                axis=0,
            )
            maximum1 = tl.max(
                tl.where(row_mask1[:, None], accumulator1, -float("inf")),
                axis=0,
            )
            maximum = tl.maximum(maximum0, maximum1)
            activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
            tl.store(
                pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
                activated,
                mask=valid_vocab,
            )


@triton.jit
def _h200_decoder_split_rows_tma_kernel(
    input_ptr,
    input_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_ism: tl.constexpr,
    stride_isk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wsm: tl.constexpr,
    stride_wsk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_WEIGHT_ONLY: tl.constexpr,
    FP8_SCALE_BLOCK_K: tl.constexpr,
    PERSISTENT: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    PROGRAM_STEP: tl.constexpr,
    OUTER_FLATTEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Triton-3.6 Hopper descriptor candidate, orthogonal to FP8/persistence."""
    input0_descriptor = tl.make_tensor_descriptor(
        input_ptr,
        shape=[64, reduction],
        strides=[stride_im, stride_ik],
        block_shape=[64, BLOCK_K],
    )
    input1_descriptor = tl.make_tensor_descriptor(
        input_ptr + 64 * stride_im,
        shape=[rows - 64, reduction],
        strides=[stride_im, stride_ik],
        block_shape=[32, BLOCK_K],
    )
    weight_descriptor = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[columns, reduction],
        strides=[stride_wn, stride_wk],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    pid = tl.program_id(0)
    num_vocab_blocks = tl.cdiv(columns, BLOCK_N)
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    offs_m0 = tl.arange(0, 64)
    offs_m1 = 64 + tl.arange(0, 32)

    # PERSISTENT changes only the host grid; retaining the constexpr makes the
    # compiled candidate and profiling metadata unambiguous.
    if PERSISTENT:
        first_program_n = pid
    else:
        first_program_n = pid
    for program_n in tl.range(
        first_program_n,
        num_vocab_blocks,
        PROGRAM_STEP,
        flatten=OUTER_FLATTEN,
    ):
        offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_vocab = offs_n < columns
        accumulator0 = tl.zeros((64, BLOCK_N), tl.float32)
        accumulator1 = tl.zeros((32, BLOCK_N), tl.float32)
        for start_k in tl.range(
            0,
            reduction,
            BLOCK_K,
            warp_specialize=WARP_SPECIALIZE,
        ):
            input_tile0 = input0_descriptor.load([0, start_k])
            input_tile1 = input1_descriptor.load([0, start_k])
            weight_tile = weight_descriptor.load(
                [program_n * BLOCK_N, start_k]
            ).T
            if USE_FP8:
                scale_group = start_k // FP8_SCALE_BLOCK_K
                input_scale0 = tl.load(
                    input_scale_ptr
                    + offs_m0 * stride_ism
                    + scale_group * stride_isk,
                    mask=offs_m0 < rows,
                    other=0.0,
                ).to(tl.float32)
                input_scale1 = tl.load(
                    input_scale_ptr
                    + offs_m1 * stride_ism
                    + scale_group * stride_isk,
                    mask=offs_m1 < rows,
                    other=0.0,
                ).to(tl.float32)
                weight_scale = tl.load(
                    weight_scale_ptr
                    + offs_n * stride_wsm
                    + scale_group * stride_wsk,
                    mask=valid_vocab,
                    other=0.0,
                ).to(tl.float32)
                partial0 = tl.dot(input_tile0, weight_tile)
                partial1 = tl.dot(input_tile1, weight_tile)
                accumulator0 += (
                    partial0 * input_scale0[:, None] * weight_scale[None, :]
                )
                accumulator1 += (
                    partial1 * input_scale1[:, None] * weight_scale[None, :]
                )
            elif FP8_WEIGHT_ONLY:
                scale_group = start_k // FP8_SCALE_BLOCK_K
                weight_scale = tl.load(
                    weight_scale_ptr
                    + offs_n * stride_wsm
                    + scale_group * stride_wsk,
                    mask=valid_vocab,
                    other=0.0,
                ).to(tl.float32)
                dequantized_weight = (
                    weight_tile.to(tl.float32) * weight_scale[None, :]
                ).to(tl.float16)
                accumulator0 += tl.dot(
                    input_tile0.to(tl.float16), dequantized_weight
                )
                accumulator1 += tl.dot(
                    input_tile1.to(tl.float16), dequantized_weight
                )
            else:
                accumulator0 += tl.dot(
                    input_tile0.to(tl.float16), weight_tile.to(tl.float16)
                )
                accumulator1 += tl.dot(
                    input_tile1.to(tl.float16), weight_tile.to(tl.float16)
                )

        bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
        for sequence in range(NUM_SEQUENCES):
            sequence_start = tl.sum(
                tl.where(sequence_ids < sequence, lengths, 0), axis=0
            )
            sequence_length = tl.sum(
                tl.where(sequence_ids == sequence, lengths, 0), axis=0
            )
            row_mask0 = (
                (offs_m0 >= sequence_start)
                & (offs_m0 < sequence_start + sequence_length)
                & (offs_m0 < rows)
            )
            row_mask1 = (
                (offs_m1 >= sequence_start)
                & (offs_m1 < sequence_start + sequence_length)
                & (offs_m1 < rows)
            )
            maximum0 = tl.max(
                tl.where(row_mask0[:, None], accumulator0, -float("inf")),
                axis=0,
            )
            maximum1 = tl.max(
                tl.where(row_mask1[:, None], accumulator1, -float("inf")),
                axis=0,
            )
            maximum = tl.maximum(maximum0, maximum1)
            activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
            tl.store(
                pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
                activated,
                mask=valid_vocab,
            )


@triton.jit
def _a2_decoder_full_rows_max_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    seq_lens_ptr,
    pooled_ptr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pv: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_FP16_DOT: tl.constexpr,
):
    """A2 decoder: load each vocabulary weight tile exactly once.

    The official case has only 83 token rows.  Giving one program the complete
    (power-of-two padded) row dimension avoids three repeated reads of every
    decoder weight tile, the output-zeroing launch, and the cross-program
    atomics used by the portable kernel.  The four disjoint segment maxima can
    consequently be written directly.
    """
    program_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    input_tiles = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    weight_tiles = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start_k in range(0, reduction, BLOCK_K):
        input_tile = tl.load(
            input_tiles,
            mask=(offs_m[:, None] < rows) & (start_k + offs_k[None, :] < reduction),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_tiles,
            mask=(offs_n[None, :] < columns) & (start_k + offs_k[:, None] < reduction),
            other=0.0,
        )
        if USE_FP16_DOT:
            input_tile = input_tile.to(tl.float16)
            weight_tile = weight_tile.to(tl.float16)
        accumulator += tl.dot(input_tile, weight_tile)
        input_tiles += BLOCK_K * stride_ik
        weight_tiles += BLOCK_K * stride_wk
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < NUM_SEQUENCES,
        other=0,
    ).to(tl.int32)
    valid_vocab = offs_n < columns
    bias = tl.load(bias_ptr + offs_n, mask=valid_vocab, other=0.0)
    for sequence in range(NUM_SEQUENCES):
        start = tl.sum(tl.where(sequence_ids < sequence, lengths, 0), axis=0)
        length = tl.sum(tl.where(sequence_ids == sequence, lengths, 0), axis=0)
        row_mask = (offs_m >= start) & (offs_m < start + length) & (offs_m < rows)
        maximum = tl.max(
            tl.where(row_mask[:, None], accumulator, -float("inf")), axis=0
        )
        # Bias is constant along the token axis, so max(x + b) = max(x) + b.
        # Adding it after reduction avoids broadcasting it over 128 rows.
        activated = tl.log(1.0 + tl.maximum(maximum + bias, 0.0))
        tl.store(
            pooled_ptr + sequence * stride_pb + offs_n * stride_pv,
            activated,
            mask=valid_vocab,
        )


@triton.jit
def _layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    columns: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program normalizes one complete hidden-state row."""
    row = tl.program_id(0)
    columns_offset = tl.arange(0, BLOCK_N)
    mask = columns_offset < columns
    values = tl.load(
        input_ptr + row * stride_im + columns_offset * stride_in,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    normalized = centered * tl.rsqrt(variance + eps)
    scale = tl.load(weight_ptr + columns_offset, mask=mask, other=0.0)
    shift = tl.load(bias_ptr + columns_offset, mask=mask, other=0.0)
    result = normalized * scale + shift
    tl.store(
        output_ptr + row * stride_om + columns_offset * stride_on,
        result,
        mask=mask,
    )


@triton.jit
def _layer_norm_stats_kernel(
    input_ptr,
    stats_ptr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_ss: tl.constexpr,
    columns: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Store exact FP32 LayerNorm mean/rstd without materializing its output."""
    row = tl.program_id(0)
    columns_offset = tl.arange(0, BLOCK_N)
    mask = columns_offset < columns
    values = tl.load(
        input_ptr + row * stride_im + columns_offset * stride_in,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    rstd = tl.rsqrt(variance + eps)
    tl.store(stats_ptr + row * stride_sm, mean)
    tl.store(stats_ptr + row * stride_sm + stride_ss, rstd)


@triton.jit
def _layer_norm_fp8_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    scale_ptr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sk: tl.constexpr,
    columns: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
):
    """LayerNorm plus per-row/per-K-block E4M3 activation quantisation.

    Quantising here avoids a separate activation scan and kernel launch.  The
    decoder consumes the stored scales while retaining FP32 accumulation and
    epilogue arithmetic.
    """
    tl.static_assert(BLOCK_N % SCALE_BLOCK_K == 0)
    row = tl.program_id(0)
    columns_offset = tl.arange(0, BLOCK_N)
    mask = columns_offset < columns
    values = tl.load(
        input_ptr + row * stride_im + columns_offset * stride_in,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    normalized = centered * tl.rsqrt(variance + eps)
    affine_scale = tl.load(
        weight_ptr + columns_offset, mask=mask, other=0.0
    ).to(tl.float32)
    affine_shift = tl.load(
        bias_ptr + columns_offset, mask=mask, other=0.0
    ).to(tl.float32)
    result = normalized * affine_scale + affine_shift

    scale_groups = BLOCK_N // SCALE_BLOCK_K
    grouped = tl.reshape(result, (scale_groups, SCALE_BLOCK_K))
    maximum = tl.max(tl.abs(grouped), axis=1)
    # E4M3FN's largest finite magnitude is 448.  A nonzero scale also makes
    # all-zero padded groups deterministic.
    quant_scale = tl.maximum(maximum / 448.0, 1.0e-12)
    quantized = grouped / quant_scale[:, None]
    quantized = tl.minimum(tl.maximum(quantized, -448.0), 448.0)
    quantized = tl.reshape(quantized, (BLOCK_N,))
    tl.store(
        output_ptr + row * stride_om + columns_offset * stride_on,
        quantized,
        mask=mask,
    )

    scale_group = tl.arange(0, scale_groups)
    valid_groups = scale_group < tl.cdiv(columns, SCALE_BLOCK_K)
    tl.store(
        scale_ptr + row * stride_sm + scale_group * stride_sk,
        quant_scale,
        mask=valid_groups,
    )


@triton.jit
def _variable_length_pool_kernel(
    activated_ptr,
    seq_lens_ptr,
    pooled_ptr,
    activated_stride_m: tl.constexpr,
    activated_stride_v: tl.constexpr,
    pooled_stride_b: tl.constexpr,
    pooled_stride_v: tl.constexpr,
    rows: tl.constexpr,
    vocab_size: tl.constexpr,
    num_sequences: tl.constexpr,
    IS_MAX: tl.constexpr,
    APPLY_SPLADE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_SEQUENCES: tl.constexpr,
):
    """Pool one sequence and a vocabulary tile without copying lengths to the CPU."""
    sequence = tl.program_id(0)
    vocab_block = tl.program_id(1)
    sequence_ids = tl.arange(0, BLOCK_SEQUENCES)
    lengths = tl.load(
        seq_lens_ptr + sequence_ids,
        mask=sequence_ids < num_sequences,
        other=0,
    ).to(tl.int32)
    start = tl.sum(tl.where(sequence_ids < sequence, lengths, 0), axis=0)
    length = tl.sum(tl.where(sequence_ids == sequence, lengths, 0), axis=0)

    row_offsets = tl.arange(0, BLOCK_ROWS)
    vocab_offsets = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = (
        (row_offsets >= start)
        & (row_offsets < start + length)
        & (row_offsets < rows)
    )
    vocab_mask = vocab_offsets < vocab_size
    if IS_MAX:
        values = tl.load(
            activated_ptr
            + row_offsets[:, None] * activated_stride_m
            + vocab_offsets[None, :] * activated_stride_v,
            mask=row_mask[:, None] & vocab_mask[None, :],
            other=-float("inf"),
        )
        result = tl.max(values, axis=0)
    else:
        values = tl.load(
            activated_ptr
            + row_offsets[:, None] * activated_stride_m
            + vocab_offsets[None, :] * activated_stride_v,
            mask=row_mask[:, None] & vocab_mask[None, :],
            other=0.0,
        )
        result = tl.sum(values, axis=0)

    if APPLY_SPLADE:
        result = tl.log(1.0 + tl.maximum(result, 0.0))

    tl.store(
        pooled_ptr
        + sequence * pooled_stride_b
        + vocab_offsets * pooled_stride_v,
        result,
        mask=vocab_mask,
    )


class ModelNew(nn.Module):
    """State-compatible SPLADE head with an accelerator-only Triton forward."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        # Keep the exact reference hierarchy and state_dict keys.  The modules
        # own parameters only; their PyTorch forward methods are never called.
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling
        (
            self._a2_decoder_mode,
            self._a2_decoder_experiment_config,
        ) = _get_a2_decoder_experiment()
        (
            self._linear_block_m,
            self._linear_block_n,
            self._linear_block_k,
            self._linear_group_m,
            self._linear_warps,
            self._linear_stages,
            self._layer_norm_warps,
            self._pool_block_v,
            self._pool_warps,
        ) = _get_launch_config()
        chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
        self._decoder_precision = _get_h200_decoder_precision()
        int8_residual_key = os.environ.get(
            "KERNELSWIFT_TASK04_INT8_RESIDUAL", "0"
        ).strip().lower()
        int8_residual_aliases = {
            "false": 0,
            "no": 0,
            "true": 12,
            "yes": 12,
            "1": 12,
        }
        if int8_residual_key in int8_residual_aliases:
            self._int8_residual_groups = int8_residual_aliases[
                int8_residual_key
            ]
        elif int8_residual_key.isdigit():
            self._int8_residual_groups = int(int8_residual_key)
        else:
            raise ValueError("Task04 INT8_RESIDUAL must be 0, 4, 8, or 12")
        if self._int8_residual_groups not in (0, 4, 8, 12):
            raise ValueError("Task04 INT8_RESIDUAL must be 0, 4, 8, or 12")
        self._int8_residual = self._int8_residual_groups > 0
        self._h200_hopper_mode = _get_h200_hopper_mode()
        self._h200_candidate_requested = (
            self._decoder_precision not in ("fp16", "int8_block64")
            or self._h200_hopper_mode != "pointer"
        )
        if self._decoder_precision.endswith("block64"):
            self._fp8_scale_block_k = 64
        elif self._decoder_precision.endswith("block128"):
            self._fp8_scale_block_k = 128
        else:
            self._fp8_scale_block_k = 1024
        if (
            self._decoder_precision != "fp16"
            or self._h200_hopper_mode != "pointer"
        ) and not self._a2_decoder_mode.startswith("split_"):
            raise ValueError(
                "Task04 Hopper precision/scheduling candidates require split_64p32"
            )
        self._decoder_fp16_dot = (
            os.environ.get("KERNELSWIFT_TASK04_FP16_DOT", "1").strip() == "1"
        )
        # The split decoder already consumes FP16 dot operands.  Materialising
        # that derived, non-persistent operand once removes a 30,522x768 cast
        # from every invocation (about 44.7 MiB for the official model).
        self._prepack_decoder_fp16 = _env_flag(
            "KERNELSWIFT_TASK04_PREPACK_DECODER_FP16", default=True
        )
        self._specialize_seq4 = (
            chip == "nvidia_h200"
            and _env_flag("KERNELSWIFT_TASK04_SPECIALIZE_SEQ4", default=False)
        )
        self._prepack_dense_fp16 = _env_flag(
            "KERNELSWIFT_TASK04_PREPACK_DENSE_FP16",
            default=chip == "nvidia_h200",
        )
        self._pack_intermediates = _env_flag(
            "KERNELSWIFT_TASK04_PACK_INTERMEDIATES", default=False
        )
        self._ln_decoder_mode = os.environ.get(
            "KERNELSWIFT_TASK04_LAYERNORM_DECODER", "materialize"
        ).strip().lower()
        if self._ln_decoder_mode not in ("materialize", "stats"):
            raise ValueError(
                "Task04 LAYERNORM_DECODER must be materialize or stats"
            )
        if chip != "nvidia_h200" and self._ln_decoder_mode != "materialize":
            raise ValueError("Task04 LayerNorm/decoder candidates require nvidia_h200")
        self._frontend_mode = os.environ.get(
            "KERNELSWIFT_TASK04_FRONTEND_MODE",
            "fused" if chip == "nvidia_h200" else "materialize",
        ).strip().lower()
        if self._frontend_mode not in ("materialize", "fused"):
            raise ValueError("Task04 FRONTEND_MODE must be materialize or fused")
        if chip != "nvidia_h200" and self._frontend_mode != "materialize":
            raise ValueError("Task04 fused frontend requires nvidia_h200")
        frontend_rows_text = os.environ.get(
            "KERNELSWIFT_TASK04_FRONTEND_ROWS", "4"
        ).strip()
        if frontend_rows_text not in ("4", "8", "16"):
            raise ValueError("Task04 FRONTEND_ROWS must be 4, 8, or 16")
        self._frontend_rows = int(frontend_rows_text)
        self._normalized_fp16 = (
            chip == "nvidia_h200"
            and _env_flag("KERNELSWIFT_TASK04_NORMALIZED_FP16", default=True)
        )
        launch_mode = os.environ.get(
            "KERNELSWIFT_TASK04_LAUNCH_MODE",
            "direct" if chip == "nvidia_h200" else "jit",
        ).strip().lower()
        if launch_mode not in ("jit", "direct"):
            raise ValueError("Task04 LAUNCH_MODE must be jit or direct")
        self._direct_launch_requested = (
            launch_mode == "direct"
            and chip == "nvidia_h200"
            and getattr(triton, "__version__", "") == "3.6.0"
        )
        self._direct_frontend_runner = None
        self._direct_frontend_device = None
        self._direct_decoder_runner = None
        self._direct_decoder_device = None
        self._direct_decoder_kind = None
        self._direct_decoder_dtype = None
        if self._decoder_precision != "fp16" and not hasattr(
            torch, "float8_e4m3fn"
        ):
            raise RuntimeError("Task04 FP8 candidates require torch.float8_e4m3fn")
        if self._decoder_precision != "fp16" and not (
            getattr(triton, "__version__", "") == "3.6.0"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (9, 0)
        ):
            raise RuntimeError(
                "Task04 FP8 candidates require Triton 3.6.0 and CUDA capability 9.0"
            )
        if (
            self._decoder_precision.endswith(("block64", "block128"))
            and hidden_size % self._fp8_scale_block_k != 0
        ):
            raise ValueError(
                "Task04 FP8 block scaling requires a divisible hidden_size"
            )
        wants_tma = "tma" in self._h200_hopper_mode
        self._h200_tma_ready = (
            getattr(triton, "__version__", "") == "3.6.0"
            and hasattr(triton, "set_allocator")
            and hasattr(tl, "make_tensor_descriptor")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (9, 0)
        )
        if wants_tma and not self._h200_tma_ready:
            raise RuntimeError(
                "Task04 TMA candidates require Triton 3.6.0 and CUDA capability 9.0"
            )
        if (
            self._h200_hopper_mode != "pointer"
            and self._decoder_precision == "fp16"
            and not self._decoder_fp16_dot
        ):
            raise ValueError("Task04 Hopper FP16 candidates require FP16_DOT=1")
        if wants_tma:
            triton.set_allocator(_task04_tma_allocator)
        self.register_buffer("_decoder_weight_fp16", None, persistent=False)
        self.register_buffer("_dense_weight_fp16", None, persistent=False)
        self.register_buffer("_decoder_weight_int8", None, persistent=False)
        self.register_buffer("_decoder_weight_int8_scale", None, persistent=False)
        self.register_buffer("_decoder_weight_fp8", None, persistent=False)
        self.register_buffer("_decoder_weight_fp8_scale", None, persistent=False)
        self._refresh_prepacked_weights()

        graph_key = os.environ.get("KERNELSWIFT_TASK04_GRAPH_REPLAY")
        if graph_key is None:
            graph_key = os.environ.get("KERNELSWIFT_H200_GRAPH_REPLAY", "1")
        graph_key = graph_key.strip().lower()
        if graph_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task04 GRAPH_REPLAY must be 0 or 1")
        self._use_graph_replay = (
            chip == "nvidia_h200"
            and graph_key in ("1", "true", "yes")
            and torch.cuda.is_available()
        )
        raw_graph_key = os.environ.get(
            "KERNELSWIFT_TASK04_RAW_GRAPH_LAUNCH", "1"
        ).strip().lower()
        if raw_graph_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task04 RAW_GRAPH_LAUNCH must be 0 or 1")
        self._use_raw_graph_launch = (
            self._use_graph_replay
            and raw_graph_key in ("1", "true", "yes")
            and getattr(triton, "__version__", "") == "3.6.0"
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_outputs = None
        self._raw_graph_exec = None
        self._raw_graph_device_index = None

    def _refresh_prepacked_weights(self) -> None:
        # Derived accelerator operands are deliberately non-persistent: the
        # reference-compatible FP32 Parameters remain the sole state_dict
        # source of truth and keep their original keys and dtype.
        if self._prepack_decoder_fp16:
            self._decoder_weight_fp16 = self.decoder.weight.detach().to(
                dtype=torch.float16
            )
        else:
            self._decoder_weight_fp16 = None
        if self._decoder_precision == "int8_block64":
            weight = self.decoder.weight.detach().to(dtype=torch.float32)
            weight_groups = weight.reshape(weight.shape[0], 12, 64)
            int8_scales = (
                weight_groups.abs().amax(dim=2).to(dtype=torch.float32) / 127.0
            ).clamp_min(1.0e-12)
            self._decoder_weight_int8 = (
                torch.round(weight_groups / int8_scales[:, :, None])
                .clamp(-127, 127)
                .to(dtype=torch.int8)
                .reshape_as(weight)
                .contiguous()
            )
            self._decoder_weight_int8_scale = int8_scales.contiguous()
        else:
            self._decoder_weight_int8 = None
            self._decoder_weight_int8_scale = None
        if self._decoder_precision.startswith("fp8_"):
            fp8_dtype = torch.float8_e4m3fn
            weight = self.decoder.weight.detach().to(dtype=torch.float32)
            scale_block_k = self._fp8_scale_block_k
            scale_groups = (weight.shape[1] + scale_block_k - 1) // scale_block_k
            padded_k = scale_groups * scale_block_k
            if padded_k != weight.shape[1]:
                # Only the row-scaled variant admits a non-divisible reduction;
                # its single group is already represented by the unpadded row.
                weight_groups = weight.reshape(weight.shape[0], 1, weight.shape[1])
            else:
                weight_groups = weight.reshape(
                    weight.shape[0], scale_groups, scale_block_k
                )
            scales = (
                weight_groups.abs().max(dim=2).values.to(dtype=torch.float32)
                / 448.0
            ).clamp_min(1.0e-12)
            quantized = (
                weight_groups / scales[:, :, None]
            ).clamp(-448.0, 448.0).to(dtype=fp8_dtype)
            self._decoder_weight_fp8 = quantized.reshape_as(weight).contiguous()
            self._decoder_weight_fp8_scale = scales.contiguous()
        else:
            self._decoder_weight_fp8 = None
            self._decoder_weight_fp8_scale = None
        if self._prepack_dense_fp16:
            self._dense_weight_fp16 = self.dense.weight.detach().to(
                dtype=torch.float16
            )
        else:
            self._dense_weight_fp16 = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        self._refresh_prepacked_weights()
        self._graph = None
        self._graph_inputs = None
        self._graph_outputs = None
        self._raw_graph_exec = None
        self._raw_graph_device_index = None
        return result

    def _launch_linear(
        self,
        source: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        destination: torch.Tensor,
        activation: int,
        use_fp16_dot: bool = False,
    ) -> None:
        rows, reduction = source.shape
        columns = weight.shape[0]
        block_m = self._linear_block_m
        block_n = self._linear_block_n
        block_k = self._linear_block_k
        grid = (
            triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),
        )
        _linear_activation_kernel[grid](
            source,
            weight,
            bias,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            ACTIVATION=activation,
            USE_FP16_DOT=use_fp16_dot,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=self._linear_group_m,
            num_warps=self._linear_warps,
            num_stages=self._linear_stages,
        )

    def _launch_decoder_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
        use_fp16: bool = False,
    ) -> None:
        rows, reduction = source.shape
        weight = (
            self._decoder_weight_fp16
            if use_fp16 and self._decoder_weight_fp16 is not None
            else self.decoder.weight
        )
        columns = weight.shape[0]
        block_m = self._linear_block_m
        block_n = self._linear_block_n
        block_k = self._linear_block_k
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        grid = (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)
        _decoder_max_pool_kernel[grid](
            source,
            weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=block_sequences,
            USE_FP16_DOT=use_fp16,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=self._linear_group_m,
            num_warps=self._linear_warps,
            num_stages=self._linear_stages,
        )

    def _launch_decoder_vocab_owner_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        columns = self.decoder.weight.shape[0]
        block_m, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        vocab_blocks = triton.cdiv(columns, block_n)
        if "_persistent_" in self._a2_decoder_mode:
            device_index = (
                source.device.index if source.device.index is not None else 0
            )
            grid = (
                min(vocab_blocks, _ascend_core_count(device_index)),
            )
        else:
            grid = (vocab_blocks,)
        _decoder_vocab_owner_max_pool_kernel[grid](
            source,
            self.decoder.weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            self.decoder.weight.stride(0),
            self.decoder.weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=block_sequences,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_decoder_partial_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
        partial: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        columns = self.decoder.weight.shape[0]
        block_m, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        programs_m = triton.cdiv(rows, block_m)
        grid = (programs_m * triton.cdiv(columns, block_n),)
        _decoder_partial_max_kernel[grid](
            source,
            self.decoder.weight,
            seq_lens,
            partial,
            source.stride(0),
            source.stride(1),
            self.decoder.weight.stride(0),
            self.decoder.weight.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=block_sequences,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=self._linear_group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        block_v = 256
        finalize_grid = (
            num_sequences,
            triton.cdiv(columns, block_v),
        )
        _decoder_finalize_partial_max_kernel[finalize_grid](
            partial,
            self.decoder.bias,
            destination,
            destination.stride(0),
            destination.stride(1),
            columns,
            NUM_SEQUENCES=num_sequences,
            PROGRAMS_M=programs_m,
            BLOCK_PROGRAMS_M=triton.next_power_of_2(programs_m),
            BLOCK_V=block_v,
            num_warps=4,
        )

    def _launch_a2_decoder_split_rows_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        use_prepacked = self._prepack_decoder_fp16 and self._decoder_fp16_dot
        weight = (
            self._decoder_weight_fp16
            if use_prepacked
            else self.decoder.weight
        )
        columns = weight.shape[0]
        _, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        grid = (triton.cdiv(columns, block_n),)
        use_direct_launch = (
            self._direct_launch_requested
            and rows == 83
            and reduction == 768
            and columns == 30522
            and num_sequences == 4
            and use_prepacked
            and source.shape == (83, 768)
            and source.dtype in (torch.float16, torch.float32)
            and seq_lens.dtype == torch.int32
            and destination.shape == (4, 30522)
            and destination.dtype == torch.float32
            and source.device == seq_lens.device == destination.device
            and source.is_contiguous()
            and weight.is_contiguous()
            and self.decoder.bias.is_contiguous()
            and seq_lens.is_contiguous()
            and destination.is_contiguous()
        )
        if use_direct_launch:
            signature_args = (
                source,
                source,
                source,
                source,
                weight,
                weight,
                self.decoder.bias,
                seq_lens,
                destination,
                source.stride(0),
                source.stride(1),
                weight.stride(0),
                weight.stride(1),
                destination.stride(0),
                destination.stride(1),
                rows,
                columns,
                reduction,
                num_sequences,
                block_sequences,
                self._decoder_fp16_dot,
                False,
                False,
                block_n,
                block_k,
                self._specialize_seq4,
            )
            if (
                self._direct_decoder_runner is not None
                and self._direct_decoder_kind == "split"
                and source.device == self._direct_decoder_device
                and source.dtype == self._direct_decoder_dtype
            ):
                self._direct_decoder_runner(*signature_args)
            else:
                compiled = _a2_decoder_split_rows_max_pool_kernel[grid](
                    *signature_args,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                self._direct_decoder_runner = compiled[(grid[0], 1, 1)]
                self._direct_decoder_device = source.device
                self._direct_decoder_kind = "split"
                self._direct_decoder_dtype = source.dtype
            return
        _a2_decoder_split_rows_max_pool_kernel[grid](
            source,
            source,
            source,
            source,
            weight,
            weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=block_sequences,
            USE_FP16_DOT=self._decoder_fp16_dot,
            USE_INT8_WEIGHT=False,
            INT8_RESIDUAL_GROUPS=0,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPECIALIZE_SEQ4=(self._specialize_seq4 and num_sequences == 4),
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_h200_decoder_nphase_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        weight = self._decoder_weight_fp16
        if weight is None:
            raise RuntimeError("Task04 N-phase candidate requires FP16 weights")
        columns = weight.shape[0]
        _, outer_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        inner_n = (
            32 if self._a2_decoder_mode.endswith("nphase32") else 64
        )
        if outer_n % inner_n != 0:
            raise RuntimeError("Task04 N-phase geometry must divide OUTER_N")
        num_sequences = seq_lens.numel()
        grid = (triton.cdiv(columns, outer_n),)
        use_direct_launch = (
            self._direct_launch_requested
            and rows == 83
            and reduction == 768
            and columns == 30522
            and num_sequences == 4
            and outer_n in (64, 128, 256)
            and inner_n in (32, 64)
            and source.shape == (83, 768)
            and source.dtype == torch.float32
            and weight.dtype == torch.float16
            and seq_lens.dtype == torch.int32
            and destination.shape == (4, 30522)
            and destination.dtype == torch.float32
            and source.device == seq_lens.device == destination.device
            and source.is_contiguous()
            and weight.is_contiguous()
            and self.decoder.bias.is_contiguous()
            and seq_lens.is_contiguous()
            and destination.is_contiguous()
        )
        signature_args = (
            source,
            weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            num_sequences,
            triton.next_power_of_2(num_sequences),
            outer_n,
            inner_n,
            outer_n // inner_n,
            block_k,
        )
        if use_direct_launch:
            if (
                self._direct_decoder_runner is not None
                and self._direct_decoder_kind == "nphase"
                and source.device == self._direct_decoder_device
                and source.dtype == self._direct_decoder_dtype
            ):
                self._direct_decoder_runner(*signature_args)
            else:
                compiled = _h200_decoder_nphase_max_pool_kernel[grid](
                    *signature_args,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                self._direct_decoder_runner = compiled[(grid[0], 1, 1)]
                self._direct_decoder_device = source.device
                self._direct_decoder_kind = "nphase"
                self._direct_decoder_dtype = source.dtype
            return
        _h200_decoder_nphase_max_pool_kernel[grid](
            *signature_args,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_h200_decoder_int8_max_pool(
        self,
        source: torch.Tensor,
        input_scale: torch.Tensor,
        input_residual: torch.Tensor,
        input_residual_scale: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        """Run the split decoder with block-scaled INT8 activations and weights."""
        weight = self._decoder_weight_int8
        weight_scale = self._decoder_weight_int8_scale
        if weight is None or weight_scale is None:
            raise RuntimeError("Task04 INT8 derived weights are not initialized")
        rows, reduction = source.shape
        columns = weight.shape[0]
        _, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        grid = (triton.cdiv(columns, block_n),)
        signature_args = (
            source,
            input_scale,
            input_residual,
            input_residual_scale,
            weight,
            weight_scale,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            num_sequences,
            block_sequences,
            True,
            True,
            self._int8_residual_groups,
            block_n,
            block_k,
            self._specialize_seq4,
        )
        use_direct_launch = (
            self._direct_launch_requested
            and rows == 83
            and reduction == 768
            and columns == 30522
            and num_sequences == 4
            and source.dtype == torch.int8
            and input_scale.shape == (83, 12)
            and input_scale.dtype == torch.float32
            and input_residual.shape == (83, 768)
            and input_residual.dtype == torch.int8
            and input_residual_scale.shape == (83, 12)
            and input_residual_scale.dtype == torch.float32
            and weight.dtype == torch.int8
            and weight_scale.shape == (30522, 12)
            and weight_scale.dtype == torch.float32
            and destination.shape == (4, 30522)
            and destination.dtype == torch.float32
            and source.device
            == input_scale.device
            == input_residual.device
            == input_residual_scale.device
            == weight.device
            == weight_scale.device
            == seq_lens.device
            == destination.device
            and source.is_contiguous()
            and input_scale.is_contiguous()
            and input_residual.is_contiguous()
            and input_residual_scale.is_contiguous()
            and weight.is_contiguous()
            and weight_scale.is_contiguous()
            and seq_lens.is_contiguous()
            and destination.is_contiguous()
        )
        if use_direct_launch:
            if (
                self._direct_decoder_runner is not None
                and self._direct_decoder_kind == "int8"
                and source.device == self._direct_decoder_device
                and source.dtype == self._direct_decoder_dtype
            ):
                self._direct_decoder_runner(*signature_args)
            else:
                compiled = _a2_decoder_split_rows_max_pool_kernel[grid](
                    *signature_args,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                self._direct_decoder_runner = compiled[(grid[0], 1, 1)]
                self._direct_decoder_device = source.device
                self._direct_decoder_kind = "int8"
                self._direct_decoder_dtype = source.dtype
            return
        _a2_decoder_split_rows_max_pool_kernel[grid](
            *signature_args,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_h200_decoder_ln_stats_max_pool(
        self,
        source: torch.Tensor,
        stats: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        weight = self._decoder_weight_fp16
        if weight is None:
            raise RuntimeError("Task04 LayerNorm-stats candidate requires FP16 weights")
        columns = weight.shape[0]
        _, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        grid = (triton.cdiv(columns, block_n),)
        _h200_decoder_ln_stats_max_pool_kernel[grid](
            source,
            stats,
            self.layer_norm.weight,
            self.layer_norm.bias,
            weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            stats.stride(0),
            stats.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=triton.next_power_of_2(num_sequences),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_h200_decoder_sequence_owner_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        weight = (
            self._decoder_weight_fp16
            if self._prepack_decoder_fp16
            else self.decoder.weight
        )
        columns = weight.shape[0]
        block_m, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        grid = (num_sequences, triton.cdiv(columns, block_n))
        _h200_decoder_sequence_owner_max_pool_kernel[grid](
            source,
            weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=triton.next_power_of_2(num_sequences),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_h200_decoder_sequential_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        weight = self._decoder_weight_fp16
        if weight is None:
            raise RuntimeError("Task04 sequential decoder requires FP16 weights")
        columns = weight.shape[0]
        _, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        grid = (triton.cdiv(columns, block_n),)
        _h200_decoder_sequential_64p32_max_pool_kernel[grid](
            source,
            weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=triton.next_power_of_2(num_sequences),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_h200_decoder_candidate(
        self,
        source: torch.Tensor,
        source_scale: torch.Tensor | None,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        """Launch the opt-in FP8/persistent/TMA decoder experiment."""
        rows, reduction = source.shape
        use_fp8 = self._decoder_precision in (
            "fp8_row",
            "fp8_block128",
            "fp8_block64",
        )
        fp8_weight_only = self._decoder_precision.startswith("fp8_weight_")
        if use_fp8 or fp8_weight_only:
            weight = self._decoder_weight_fp8
            weight_scale = self._decoder_weight_fp8_scale
            if weight is None or weight_scale is None:
                raise RuntimeError("Task04 FP8 derived operands are not initialized")
            if use_fp8:
                if source_scale is None:
                    raise RuntimeError("Task04 FP8 activation scales are not initialized")
                input_scale = source_scale
            else:
                input_scale = source
        else:
            weight = (
                self._decoder_weight_fp16
                if self._prepack_decoder_fp16
                else self.decoder.weight
            )
            # The scale arguments are compiled away for FP16; valid tensor
            # placeholders keep the launch ABI uniform across the matrix.
            input_scale = source
            weight_scale = weight

        columns = weight.shape[0]
        _, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        vocab_blocks = triton.cdiv(columns, block_n)
        persistent = self._h200_hopper_mode.startswith("persistent")
        if persistent:
            device_index = (
                source.device.index if source.device.index is not None else 0
            )
            grid = (min(vocab_blocks, _h200_sm_count(device_index)),)
        else:
            grid = (vocab_blocks,)

        common = (
            source,
            input_scale,
            weight,
            weight_scale,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            input_scale.stride(0),
            input_scale.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
        )
        if "tma" in self._h200_hopper_mode:
            _h200_decoder_split_rows_tma_kernel[grid](
                *common,
                NUM_SEQUENCES=num_sequences,
                BLOCK_SEQUENCES=block_sequences,
                USE_FP8=use_fp8,
                FP8_WEIGHT_ONLY=fp8_weight_only,
                FP8_SCALE_BLOCK_K=self._fp8_scale_block_k,
                PERSISTENT=persistent,
                WARP_SPECIALIZE=self._h200_hopper_mode.endswith("_ws"),
                PROGRAM_STEP=grid[0],
                OUTER_FLATTEN=not (
                    persistent and self._h200_hopper_mode.endswith("_ws")
                ),
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            _h200_decoder_split_rows_persistent_kernel[grid](
                *common,
                NUM_SEQUENCES=num_sequences,
                BLOCK_SEQUENCES=block_sequences,
                USE_FP8=use_fp8,
                FP8_WEIGHT_ONLY=fp8_weight_only,
                FP8_SCALE_BLOCK_K=self._fp8_scale_block_k,
                PROGRAM_STEP=grid[0],
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    def _launch_decoder_raw_logits(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        weight = (
            self._decoder_weight_fp16
            if self._prepack_decoder_fp16
            else self.decoder.weight
        )
        columns = weight.shape[0]
        block_m, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        grid = (
            triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),
        )
        _linear_activation_kernel[grid](
            source,
            weight,
            self.decoder.bias,
            destination,
            source.stride(0),
            source.stride(1),
            weight.stride(0),
            weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            ACTIVATION=0,
            USE_FP16_DOT=self._prepack_decoder_fp16,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=self._linear_group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_a2_decoder_full_rows_max_pool(
        self,
        source: torch.Tensor,
        seq_lens: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        rows, reduction = source.shape
        columns = self.decoder.weight.shape[0]
        # Full-row modes use the first tuple element as their explicitly
        # audited padded-row bound.  The official 83-token case uses 128.
        block_m, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        grid = (triton.cdiv(columns, block_n),)
        _a2_decoder_full_rows_max_pool_kernel[grid](
            source,
            self.decoder.weight,
            self.decoder.bias,
            seq_lens,
            destination,
            source.stride(0),
            source.stride(1),
            self.decoder.weight.stride(0),
            self.decoder.weight.stride(1),
            destination.stride(0),
            destination.stride(1),
            rows,
            columns,
            reduction,
            NUM_SEQUENCES=num_sequences,
            BLOCK_SEQUENCES=block_sequences,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            # FP32 is intentional: direct FP16 casts were slower on the first
            # 910B2C measurements.  This experiment isolates ownership/layout
            # from operand-precision changes.
            USE_FP16_DOT=False,
            num_warps=num_warps,
            # A single software stage avoids A2 multi-buffer UB pressure; the
            # 64-wide K tile already amortises loop/control overhead.
            num_stages=num_stages,
        )

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        dense_weight = (
            self._dense_weight_fp16
            if self._prepack_dense_fp16
            else self.dense.weight
        )
        if self._decoder_precision == "int8_block64":
            decoder_weight = self._decoder_weight_int8
            decoder_scale = self._decoder_weight_int8_scale
        else:
            decoder_weight = (
                self._decoder_weight_fp16
                if self._prepack_decoder_fp16
                else self.decoder.weight
            )
            # A stable placeholder keeps the graph identity tuple uniform.
            decoder_scale = decoder_weight
        graph = self._graph
        graph_inputs = self._graph_inputs
        if (
            graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is hidden_states
            and graph_inputs[1] is seq_lens
            and graph_inputs[2] is dense_weight
            and graph_inputs[3] is self.dense.bias
            and graph_inputs[4] is self.layer_norm.weight
            and graph_inputs[5] is self.layer_norm.bias
            and graph_inputs[6] is decoder_weight
            and graph_inputs[7] is decoder_scale
            and graph_inputs[8] is self.decoder.bias
        ):
            raw_graph_exec = self._raw_graph_exec
            if raw_graph_exec is None:
                graph.replay()
            else:
                stream = _cuda_runtime.cudaStream_t(
                    torch._C._cuda_getCurrentRawStream(
                        self._raw_graph_device_index
                    )
                )
                status = _cuda_runtime.cudaGraphLaunch(
                    raw_graph_exec, stream
                )[0]
                if status != _cuda_runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaGraphLaunch failed: {status}")
            return self._graph_outputs

        official = (
            self._use_graph_replay
            and self._direct_launch_requested
            and self._frontend_mode == "fused"
            and self._a2_decoder_mode.startswith("split_")
            and self._decoder_precision in ("fp16", "int8_block64")
            and self._h200_hopper_mode == "pointer"
            and self._ln_decoder_mode == "materialize"
            and self.pooling == "max"
            and hidden_states.shape == (83, 768)
            and seq_lens.shape == (4,)
            and hidden_states.dtype == torch.float32
            and seq_lens.dtype == torch.int32
            and dense_weight is not None
            and decoder_weight is not None
            and dense_weight.shape == (768, 768)
            and decoder_weight is not None
            and decoder_scale is not None
            and decoder_weight.shape == (30522, 768)
            and dense_weight.dtype == torch.float16
            and (
                (
                    self._decoder_precision == "fp16"
                    and decoder_weight.dtype == torch.float16
                    and decoder_scale is decoder_weight
                )
                or (
                    self._decoder_precision == "int8_block64"
                    and decoder_weight.dtype == torch.int8
                    and decoder_scale.shape == (30522, 12)
                    and decoder_scale.dtype == torch.float32
                )
            )
            and hidden_states.device
            == seq_lens.device
            == dense_weight.device
            == decoder_weight.device
            == decoder_scale.device
            == self.dense.bias.device
            == self.layer_norm.weight.device
            == self.layer_norm.bias.device
            == self.decoder.bias.device
            and hidden_states.device.type == "cuda"
            and hidden_states.is_contiguous()
            and seq_lens.is_contiguous()
            and dense_weight.is_contiguous()
            and decoder_weight.is_contiguous()
            and decoder_scale.is_contiguous()
            and self.dense.bias.is_contiguous()
            and self.layer_norm.weight.is_contiguous()
            and self.layer_norm.bias.is_contiguous()
            and self.decoder.bias.is_contiguous()
        )
        if not official:
            return self._forward_impl(hidden_states, seq_lens)

        self._forward_impl(hidden_states, seq_lens)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._forward_impl(hidden_states, seq_lens)
        self._graph = graph
        self._graph_inputs = (
            hidden_states,
            seq_lens,
            dense_weight,
            self.dense.bias,
            self.layer_norm.weight,
            self.layer_norm.bias,
            decoder_weight,
            decoder_scale,
            self.decoder.bias,
        )
        self._graph_outputs = outputs
        if self._use_raw_graph_launch:
            self._raw_graph_exec = _cuda_runtime.cudaGraphExec_t(
                graph.raw_cuda_graph_exec()
            )
            self._raw_graph_device_index = hidden_states.device.index
        else:
            self._raw_graph_exec = None
            self._raw_graph_device_index = None
        graph.replay()
        return outputs

    def _forward_impl(
        self, hidden_states: torch.Tensor, seq_lens: torch.Tensor
    ) -> list:
        if hidden_states.device.type == "cpu" or seq_lens.device.type == "cpu":
            raise RuntimeError("ModelNew requires accelerator tensors")
        if hidden_states.device != seq_lens.device:
            raise ValueError("hidden_states and seq_lens must be on the same device")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.dense.in_features:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if seq_lens.ndim != 1:
            raise ValueError("seq_lens must be a rank-1 tensor")

        rows = hidden_states.shape[0]
        hidden_size = self.dense.out_features
        vocab_size = self.decoder.out_features
        num_sequences = seq_lens.numel()
        if rows == 0 or num_sequences == 0:
            raise ValueError("the reference requires non-empty tokens and sequences")

        use_h200_candidate = (
            self._h200_candidate_requested
            and self.pooling == "max"
            and rows == 83
            and hidden_size == 768
            and vocab_size == 30522
            and num_sequences == 4
            and hidden_states.dtype == torch.float32
            and hidden_states.is_contiguous()
            and seq_lens.is_contiguous()
        )
        use_fp8_candidate = (
            use_h200_candidate
            and self._decoder_precision
            in ("fp8_row", "fp8_block128", "fp8_block64")
        )
        use_int8_candidate = (
            self._decoder_precision == "int8_block64"
            and self.pooling == "max"
            and rows == 83
            and hidden_size == 768
            and vocab_size == 30522
            and num_sequences == 4
            and hidden_states.dtype == torch.float32
            and hidden_states.is_contiguous()
            and seq_lens.dtype == torch.int32
            and seq_lens.is_contiguous()
            and self._decoder_weight_int8 is not None
            and self._decoder_weight_int8_scale is not None
        )
        use_fused_frontend_candidate = (
            self._frontend_mode == "fused"
            and not use_fp8_candidate
            and rows == 83
            and hidden_size == 768
            and hidden_states.dtype == torch.float32
            and hidden_states.is_contiguous()
            and self._prepack_dense_fp16
        )
        use_h200_nphase_candidate = (
            self._a2_decoder_mode
            in ("split_h200_nphase64", "split_h200_nphase32")
            and self._ln_decoder_mode == "materialize"
            and not use_h200_candidate
            and self.pooling == "max"
            and rows == 83
            and hidden_size == 768
            and vocab_size == 30522
            and num_sequences == 4
            and hidden_states.dtype == torch.float32
            and seq_lens.dtype == torch.int32
            and hidden_states.is_contiguous()
            and seq_lens.is_contiguous()
            and self.decoder.bias.is_contiguous()
            and self._prepack_decoder_fp16
            and self._decoder_fp16_dot
            and self._decoder_weight_fp16 is not None
            and self._decoder_weight_fp16.dtype == torch.float16
            and self._decoder_weight_fp16.is_contiguous()
        )
        use_ln_stats_candidate = (
            self._ln_decoder_mode == "stats"
            and not use_fused_frontend_candidate
            and not use_h200_candidate
            and self.pooling == "max"
            and self._a2_decoder_mode.startswith("split_")
            and rows == 83
            and hidden_size == 768
            and vocab_size == 30522
            and num_sequences == 4
            and hidden_states.dtype == torch.float32
            and hidden_states.is_contiguous()
            and seq_lens.is_contiguous()
            and self._prepack_decoder_fp16
            and self._decoder_fp16_dot
        )

        layer_norm_stats = None
        normalized_int8 = None
        normalized_int8_scale = None
        normalized_int8_residual = None
        normalized_int8_residual_scale = None
        if use_fused_frontend_candidate:
            if self._normalized_fp16:
                dense_output = torch.empty(
                    (rows, hidden_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                normalized = torch.empty_like(
                    dense_output, dtype=torch.float16
                )
            else:
                normalized = torch.empty(
                    (rows, hidden_size),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                dense_output = normalized
            normalized_scale = None
        elif use_ln_stats_candidate:
            intermediate_elements = rows * hidden_size
            intermediate_storage = torch.empty(
                intermediate_elements + 2 * rows,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            dense_output = intermediate_storage[:intermediate_elements].view(
                rows, hidden_size
            )
            layer_norm_stats = intermediate_storage[intermediate_elements:].view(
                rows, 2
            )
            # This placeholder is never consumed by the stats specialization;
            # assigning it keeps every generic fallback visibly initialized.
            normalized = dense_output
            normalized_scale = None
        elif self._pack_intermediates and not use_fp8_candidate:
            intermediate_elements = rows * hidden_size
            intermediate_storage = torch.empty(
                2 * intermediate_elements,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            dense_output = intermediate_storage[:intermediate_elements].view(
                rows, hidden_size
            )
            normalized = intermediate_storage[intermediate_elements:].view(
                rows, hidden_size
            )
            normalized_scale = None
        else:
            dense_output = torch.empty(
                (rows, hidden_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        if use_fp8_candidate:
            normalized = torch.empty_like(
                dense_output, dtype=torch.float8_e4m3fn
            )
            activation_scale_groups = triton.cdiv(
                hidden_size, self._fp8_scale_block_k
            )
            normalized_scale = torch.empty(
                (rows, activation_scale_groups),
                device=hidden_states.device,
                dtype=torch.float32,
            )
        elif use_int8_candidate:
            normalized_int8 = torch.empty_like(normalized, dtype=torch.int8)
            normalized_int8_scale = torch.empty(
                (rows, hidden_size // 64),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            if self._int8_residual:
                normalized_int8_residual = torch.empty_like(
                    normalized, dtype=torch.int8
                )
                normalized_int8_residual_scale = torch.empty_like(
                    normalized_int8_scale
                )
            else:
                normalized_int8_residual = normalized_int8
                normalized_int8_residual_scale = normalized_int8_scale
        elif (
            not self._pack_intermediates
            and not use_ln_stats_candidate
            and not use_fused_frontend_candidate
        ):
            # Preserve the validated default allocation call exactly.
            normalized = torch.empty_like(dense_output)
            normalized_scale = None
        pooled = torch.empty(
            (num_sequences, vocab_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        dense_weight = (
            self._dense_weight_fp16
            if self._prepack_dense_fp16
            else self.dense.weight
        )
        if use_fused_frontend_candidate:
            frontend_grid = (triton.cdiv(rows, self._frontend_rows),)
            use_direct_frontend = (
                self._direct_launch_requested
                and rows == 83
                and hidden_size == 768
                and hidden_states.shape == (83, 768)
                and hidden_states.dtype == torch.float32
                and dense_weight.shape == (768, 768)
                and dense_weight.dtype == torch.float16
                and dense_output.shape == (83, 768)
                and dense_output.dtype == torch.float32
                and normalized.shape == (83, 768)
                and normalized.dtype
                == (torch.float16 if self._normalized_fp16 else torch.float32)
                and hidden_states.device
                == dense_weight.device
                == dense_output.device
                == normalized.device
                and hidden_states.is_contiguous()
                and dense_weight.is_contiguous()
                and self.dense.bias.is_contiguous()
                and self.layer_norm.weight.is_contiguous()
                and self.layer_norm.bias.is_contiguous()
                and dense_output.is_contiguous()
                and normalized.is_contiguous()
            )
            if use_direct_frontend:
                signature_args = (
                    hidden_states,
                    dense_weight,
                    self.dense.bias,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                    dense_output,
                    normalized,
                    normalized_int8 if use_int8_candidate else normalized,
                    (
                        normalized_int8_scale
                        if use_int8_candidate
                        else normalized
                    ),
                    (
                        normalized_int8_residual
                        if use_int8_candidate
                        else normalized
                    ),
                    (
                        normalized_int8_residual_scale
                        if use_int8_candidate
                        else normalized
                    ),
                    hidden_states.stride(0),
                    hidden_states.stride(1),
                    dense_weight.stride(0),
                    dense_weight.stride(1),
                    dense_output.stride(0),
                    dense_output.stride(1),
                    normalized.stride(0),
                    normalized.stride(1),
                    (
                        normalized_int8.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    (
                        normalized_int8_scale.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8_scale.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    (
                        normalized_int8_residual.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8_residual.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    (
                        normalized_int8_residual_scale.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8_residual_scale.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    rows,
                    hidden_size,
                    hidden_states.shape[1],
                    self.layer_norm.eps,
                    self._frontend_rows,
                    self._linear_block_n,
                    self._linear_block_k,
                    use_int8_candidate,
                    (
                        self._int8_residual_groups
                        if use_int8_candidate
                        else 0
                    ),
                )
                if (
                    self._direct_frontend_runner is not None
                    and hidden_states.device == self._direct_frontend_device
                ):
                    self._direct_frontend_runner(*signature_args)
                else:
                    compiled = _h200_dense_gelu_layer_norm_kernel[
                        frontend_grid
                    ](
                        *signature_args,
                        num_warps=self._linear_warps,
                        num_stages=self._linear_stages,
                    )
                    self._direct_frontend_runner = compiled[
                        (frontend_grid[0], 1, 1)
                    ]
                    self._direct_frontend_device = hidden_states.device
            else:
                _h200_dense_gelu_layer_norm_kernel[frontend_grid](
                    hidden_states,
                    dense_weight,
                    self.dense.bias,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                    dense_output,
                    normalized,
                    normalized_int8 if use_int8_candidate else normalized,
                    (
                        normalized_int8_scale
                        if use_int8_candidate
                        else normalized
                    ),
                    (
                        normalized_int8_residual
                        if use_int8_candidate
                        else normalized
                    ),
                    (
                        normalized_int8_residual_scale
                        if use_int8_candidate
                        else normalized
                    ),
                    hidden_states.stride(0),
                    hidden_states.stride(1),
                    dense_weight.stride(0),
                    dense_weight.stride(1),
                    dense_output.stride(0),
                    dense_output.stride(1),
                    normalized.stride(0),
                    normalized.stride(1),
                    (
                        normalized_int8.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    (
                        normalized_int8_scale.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8_scale.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    (
                        normalized_int8_residual.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8_residual.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    (
                        normalized_int8_residual_scale.stride(0)
                        if use_int8_candidate
                        else normalized.stride(0)
                    ),
                    (
                        normalized_int8_residual_scale.stride(1)
                        if use_int8_candidate
                        else normalized.stride(1)
                    ),
                    rows,
                    hidden_size,
                    hidden_states.shape[1],
                    self.layer_norm.eps,
                    ROWS_PER_PROGRAM=self._frontend_rows,
                    BLOCK_N=self._linear_block_n,
                    BLOCK_K=self._linear_block_k,
                    QUANTIZE_INT8=use_int8_candidate,
                    INT8_RESIDUAL_GROUPS=(
                        self._int8_residual_groups
                        if use_int8_candidate
                        else 0
                    ),
                    num_warps=self._linear_warps,
                    num_stages=self._linear_stages,
                )
        else:
            self._launch_linear(
                hidden_states,
                dense_weight,
                self.dense.bias,
                dense_output,
                activation=1,
                use_fp16_dot=self._prepack_dense_fp16,
            )
        block_hidden = triton.next_power_of_2(hidden_size)
        if use_fused_frontend_candidate:
            pass
        elif use_fp8_candidate:
            _layer_norm_fp8_kernel[(rows,)](
                dense_output,
                self.layer_norm.weight,
                self.layer_norm.bias,
                normalized,
                normalized_scale,
                dense_output.stride(0),
                dense_output.stride(1),
                normalized.stride(0),
                normalized.stride(1),
                normalized_scale.stride(0),
                normalized_scale.stride(1),
                hidden_size,
                self.layer_norm.eps,
                BLOCK_N=block_hidden,
                SCALE_BLOCK_K=self._fp8_scale_block_k,
                num_warps=self._layer_norm_warps,
            )
        elif use_ln_stats_candidate:
            _layer_norm_stats_kernel[(rows,)](
                dense_output,
                layer_norm_stats,
                dense_output.stride(0),
                dense_output.stride(1),
                layer_norm_stats.stride(0),
                layer_norm_stats.stride(1),
                hidden_size,
                self.layer_norm.eps,
                BLOCK_N=block_hidden,
                num_warps=self._layer_norm_warps,
            )
        else:
            _layer_norm_kernel[(rows,)](
                dense_output,
                self.layer_norm.weight,
                self.layer_norm.bias,
                normalized,
                dense_output.stride(0),
                dense_output.stride(1),
                normalized.stride(0),
                normalized.stride(1),
                hidden_size,
                self.layer_norm.eps,
                BLOCK_N=block_hidden,
                num_warps=self._layer_norm_warps,
            )
        if self.pooling == "max":
            # The direct-write A2 path is bounded to 128 rows to keep its
            # complete-row accumulator inside UB.  Larger/unusual inputs still
            # use the portable all-custom-kernel atomic implementation.
            if use_int8_candidate:
                self._launch_h200_decoder_int8_max_pool(
                    normalized_int8,
                    normalized_int8_scale,
                    normalized_int8_residual,
                    normalized_int8_residual_scale,
                    seq_lens,
                    pooled,
                )
            elif use_ln_stats_candidate:
                self._launch_h200_decoder_ln_stats_max_pool(
                    dense_output, layer_norm_stats, seq_lens, pooled
                )
            elif use_h200_candidate:
                self._launch_h200_decoder_candidate(
                    normalized, normalized_scale, seq_lens, pooled
                )
            elif self._a2_decoder_mode.startswith("sequence_") and rows <= 128:
                self._launch_h200_decoder_sequence_owner_max_pool(
                    normalized, seq_lens, pooled
                )
            elif self._a2_decoder_mode.startswith("sequential_") and rows <= 96:
                self._launch_h200_decoder_sequential_max_pool(
                    normalized, seq_lens, pooled
                )
            elif use_h200_nphase_candidate:
                self._launch_h200_decoder_nphase_max_pool(
                    normalized, seq_lens, pooled
                )
            elif self._a2_decoder_mode.startswith("raw_"):
                raw_logits = torch.empty(
                    (rows, vocab_size),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                self._launch_decoder_raw_logits(normalized, raw_logits)
                raw_pool_block = 32
                raw_pool_grid = (
                    num_sequences,
                    triton.cdiv(vocab_size, raw_pool_block),
                )
                _variable_length_pool_kernel[raw_pool_grid](
                    raw_logits,
                    seq_lens,
                    pooled,
                    raw_logits.stride(0),
                    raw_logits.stride(1),
                    pooled.stride(0),
                    pooled.stride(1),
                    rows,
                    vocab_size,
                    num_sequences,
                    IS_MAX=True,
                    APPLY_SPLADE=True,
                    BLOCK_ROWS=triton.next_power_of_2(rows),
                    BLOCK_V=raw_pool_block,
                    BLOCK_SEQUENCES=triton.next_power_of_2(num_sequences),
                    num_warps=4,
                )
            elif self._a2_decoder_mode.startswith("split_") and rows <= 96:
                self._launch_a2_decoder_split_rows_max_pool(
                    normalized, seq_lens, pooled
                )
            elif self._a2_decoder_mode.startswith("partial_"):
                partial_rows = triton.cdiv(
                    rows, self._a2_decoder_experiment_config[0]
                )
                partial = torch.empty(
                    (partial_rows, num_sequences, vocab_size),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                self._launch_decoder_partial_max_pool(
                    normalized, seq_lens, pooled, partial
                )
            elif self._a2_decoder_mode.startswith("owner_"):
                self._launch_decoder_vocab_owner_max_pool(
                    normalized, seq_lens, pooled
                )
            elif self._a2_decoder_mode.startswith("full_") and rows <= 128:
                self._launch_a2_decoder_full_rows_max_pool(
                    normalized, seq_lens, pooled
                )
            else:
                pooled_total = num_sequences * vocab_size
                _zero_kernel[(triton.cdiv(pooled_total, 256),)](
                    pooled, pooled_total, BLOCK=256, num_warps=4
                )
                self._launch_decoder_max_pool(
                    normalized,
                    seq_lens,
                    pooled,
                    use_fp16=self._a2_decoder_mode.startswith("atomic_fp16"),
                )
        else:
            activated_logits = torch.empty(
                (rows, vocab_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            decoder_weight = (
                self._decoder_weight_fp16
                if self._prepack_decoder_fp16
                else self.decoder.weight
            )
            self._launch_linear(
                normalized,
                decoder_weight,
                self.decoder.bias,
                activated_logits,
                activation=2,
                use_fp16_dot=self._prepack_decoder_fp16,
            )
            block_rows = triton.next_power_of_2(rows)
            block_sequences = triton.next_power_of_2(num_sequences)
            block_vocab = self._pool_block_v
            pool_grid = (num_sequences, triton.cdiv(vocab_size, block_vocab))
            _variable_length_pool_kernel[pool_grid](
                activated_logits,
                seq_lens,
                pooled,
                activated_logits.stride(0),
                activated_logits.stride(1),
                pooled.stride(0),
                pooled.stride(1),
                rows,
                vocab_size,
                num_sequences,
                IS_MAX=False,
                APPLY_SPLADE=False,
                BLOCK_ROWS=block_rows,
                BLOCK_V=block_vocab,
                BLOCK_SEQUENCES=block_sequences,
                num_warps=self._pool_warps,
            )
        # ``unbind`` only creates views; all numerical work above is performed
        # by custom kernels.  A Python list is required by the reference API.
        return list(pooled.unbind(0))


class Model(ModelNew):
    """Compatibility name matching the reference module."""

    pass


def get_inputs():
    """Official Task04 benchmark inputs (also required by auto_bench.py)."""
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cuda")
    hidden_states = torch.randn(83, 768, device="cuda")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]
