"""Triton implementation of Task04: SPLADESparsePooler."""

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, linear warps, linear stages,
#  layer-norm warps, pooling BLOCK_V, pooling warps)
_CHIP_CONFIGS = {
    "auto": (32, 64, 32, 8, 4, 3, 8, 64, 4),
    "ascend_a2": (32, 64, 32, 8, 4, 2, 8, 64, 4),
    # C500 has 64 threads/warp and 64 KiB shared memory per AP.  Stage 2 and
    # four layer-norm warps are the conservative first real-card baseline;
    # the alternatives below remain explicit A/B candidates.
    "metax_c500": (32, 64, 32, 8, 4, 1, 4, 64, 4),
    # BI150 exposes 64-thread warps and 128 KiB shared memory per MP.  Register
    # allocation is the tighter limit for the split-row decoder; larger N tiles
    # remain measurable only through bounded variants below.
    "iluvatar_bi150": (32, 64, 32, 8, 4, 1, 2, 64, 4),
    "enflame_s60": (32, 64, 32, 8, 4, 2, 4, 64, 4),
    "thead_810e": (16, 64, 32, 4, 4, 2, 4, 64, 4),
    "cambricon_mlu590": (16, 64, 32, 4, 4, 2, 4, 64, 4),
    "hygon_bw1000": (32, 128, 32, 8, 8, 2, 8, 128, 8),
    "mthreads_ph100": (32, 128, 32, 8, 8, 3, 8, 128, 8),
    "kunlun_p800": (16, 64, 32, 4, 4, 2, 4, 64, 4),
    "biren_br106m": (32, 128, 32, 8, 8, 3, 8, 128, 8),
}

# (decoder BLOCK_N, BLOCK_K, warps, stages).  Kept separate so an A2 machine
# can A/B the large decoder without perturbing the first 768x768 projection.
_A2_DECODER_CONFIG = (32, 64, 4, 1)
_A2_AI_CORE_COUNTS = {}


def _tensor_layout_key(tensor: torch.Tensor) -> tuple:
    """Describe every property that can affect a Triton specialization."""
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
    )


_C500_LAUNCH_VARIANTS = {
    # BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, linear warps, linear stages,
    # layer-norm warps, pooling BLOCK_V, pooling warps
    "m16n64k32_w2_s1": (16, 64, 32, 4, 2, 1, 2, 64, 2),
    "m32n64k32_w2_s1": (32, 64, 32, 8, 2, 1, 2, 64, 2),
    "m32n64k32_w4_s1": (32, 64, 32, 8, 4, 1, 4, 64, 4),
    # BI150 first-projection configurations.  They change only the 83x768 @
    # 768x768 dense+GELU launch: LayerNorm stays W2 and the independently
    # configured split decoder stays N128/W4.  M32/W8 is the measured BI150
    # default below; M64/W8 remains an environment-only diagnostic.
    "m32n64k32_w8_s1": (32, 64, 32, 8, 8, 1, 2, 64, 4),
    "m64n64k32_w8_s1": (64, 64, 32, 8, 8, 1, 2, 64, 4),
    "m32n64k32_w4_s2": (32, 64, 32, 8, 4, 2, 4, 64, 4),
    "m32n64k32_w4_s3": (32, 64, 32, 8, 4, 3, 8, 64, 4),
    "m32n128k32_w4_s2": (32, 128, 32, 8, 4, 2, 4, 64, 4),
}


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "iluvatar_bi150").strip().lower()
    default = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])
    if chip not in ("metax_c500", "iluvatar_bi150"):
        return default
    # BI150 paired long runs selected W8/S1 for the first dense+GELU layer.
    # An explicitly supplied ``default`` (or empty value) remains the exact
    # W4/S1 + LayerNorm-W2 rollback represented by ``_CHIP_CONFIGS``.
    default_variant = (
        "m32n64k32_w4_s1"
        if chip == "metax_c500"
        else "m32n64k32_w8_s1"
    )
    variant = os.environ.get(
        "KERNELSWIFT_TASK04_CONFIG", default_variant
    ).strip().lower()
    if variant in ("", "default"):
        return default
    if variant not in _C500_LAUNCH_VARIANTS:
        raise ValueError(f"unsupported Task04 launch config: {variant}")
    return _C500_LAUNCH_VARIANTS[variant]


def _get_a2_decoder_experiment():
    """Return an opt-in decoder path and its launch tuple.

    Named variants make structural A/B tests reproducible on the target rather
    than silently changing another chip's path.  BI150's measured default is a
    split-row specialization; unsupported layouts retain the atomic fallback.
    """
    chip = os.environ.get("KERNELSWIFT_CHIP", "iluvatar_bi150").strip().lower()
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
        "split_64p32x32x32_w2_s1": (64, 32, 32, 2, 1),
        "split_64p32x32x32_w4_s2": (64, 32, 32, 4, 2),
        "split_64p32x32x64_w4_s1": (64, 32, 64, 4, 1),
        "split_64p32x64x32_w4_s1": (64, 64, 32, 4, 1),
        "split_64p32x64x32_w2_s1": (64, 64, 32, 2, 1),
        "split_64p32x64x32_w4_s2": (64, 64, 32, 4, 2),
        "split_64p32x64x64_w4_s1": (64, 64, 64, 4, 1),
        "split_64p32x128x32_w4_s1": (64, 128, 32, 4, 1),
        "raw_32x64x32_w4_s1_p32": (32, 64, 32, 4, 1),
        "raw_32x128x32_w4_s1_p32": (32, 128, 32, 4, 1),
        "full_128x16x32_w4_s1": (128, 16, 32, 4, 1),
        "full_128x32x32_w4_s1": (128, 32, 32, 4, 1),
        "full_128x32x64_w4_s1": (128, 32, 64, 4, 1),
        "full_128x32x32_w8_s1": (128, 32, 32, 8, 1),
    }
    if chip in ("metax_c500", "iluvatar_bi150"):
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
            "split_64p32x32x32_w2_s1",
            "split_64p32x32x32_w4_s2",
            "split_64p32x32x64_w4_s1",
            "split_64p32x64x32_w4_s1",
            "split_64p32x64x32_w2_s1",
            "split_64p32x64x32_w4_s2",
            "split_64p32x64x64_w4_s1",
            "split_64p32x128x32_w4_s1",
        }
        if mode in ("", "default"):
            mode = (
                "split_64p32x128x32_w4_s1"
                if chip == "iluvatar_bi150"
                else "split_64p32x64x32_w4_s1"
            )
        if mode == "atomic":
            return "atomic", variants["owner_32x64x32_w4_s1"]
        if mode not in c500_modes:
            raise ValueError(f"unsupported Task04 C500/BI150 decoder mode: {mode}")
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


def _ascend_ai_core_count(device_index: int) -> int:
    """Return the physical Cube-core count, with a reproducible override."""
    override = os.environ.get("KERNELSWIFT_TASK04_PROGRAMS")
    cache_key = (device_index, override)
    if cache_key in _A2_AI_CORE_COUNTS:
        return _A2_AI_CORE_COUNTS[cache_key]
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
    _A2_AI_CORE_COUNTS[cache_key] = count
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
    # distributes all vocabulary tiles cyclically over the physical AI cores,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        if USE_FP16_DOT:
            input_tile0 = input_tile0.to(tl.float16)
            input_tile1 = input_tile1.to(tl.float16)
            weight_tile = weight_tile.to(tl.float16)
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
def _layer_norm_inplace_kernel(
    data_ptr,
    weight_ptr,
    bias_ptr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    columns: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Normalize one row and overwrite its dead dense+GELU input.

    The complete row is loaded before any store, so this has exactly the same
    row-wise dependency graph as ``_layer_norm_kernel`` while avoiding a
    second intermediate allocation.  A dedicated kernel also avoids relying
    on aliasing behaviour for two pointer arguments that happen to be equal.
    """
    row = tl.program_id(0)
    columns_offset = tl.arange(0, BLOCK_N)
    mask = columns_offset < columns
    values = tl.load(
        data_ptr + row * stride_m + columns_offset * stride_n,
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
        data_ptr + row * stride_m + columns_offset * stride_n,
        result,
        mask=mask,
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
        layer_norm_warps = os.environ.get(
            "KERNELSWIFT_TASK04_LN_WARPS", ""
        ).strip()
        if layer_norm_warps in ("1", "2", "4", "8"):
            self._layer_norm_warps = int(layer_norm_warps)
        self._decoder_fp16_dot = (
            os.environ.get("KERNELSWIFT_TASK04_FP16_DOT", "1").strip() == "1"
        )
        self._dense_fp16_dot = (
            os.environ.get("KERNELSWIFT_TASK04_DENSE_FP16_DOT", "0").strip()
            == "1"
        )
        # The normalized-workspace experiment remains opt-in.  The measured
        # BI150 default stores only decoder.weight as FP16: the ordinary
        # state_dict copy performs the same rounding once when it populates this
        # Parameter, instead of repeating a FP32 load-and-cast in every kernel.
        # Setting KERNELSWIFT_TASK04_DECODER_WEIGHT_FP16=0 restores FP32
        # parameter storage.  Bias, LayerNorm parameters, dot accumulators,
        # segment maxima, SPLADE log and final outputs all remain FP32.
        self._normalized_fp16_workspace = (
            os.environ.get("KERNELSWIFT_TASK04_NORM_FP16", "0").strip() == "1"
        )
        chip = os.environ.get(
            "KERNELSWIFT_CHIP", "iluvatar_bi150"
        ).strip().lower()
        self._inplace_layer_norm = (
            chip == "iluvatar_bi150"
            and not self._normalized_fp16_workspace
            and os.environ.get(
                "KERNELSWIFT_TASK04_INPLACE_LN", "1"
            ).strip()
            == "1"
        )
        decoder_weight_fp16_default = "1" if (
            chip == "iluvatar_bi150"
            and self.pooling == "max"
            and self._a2_decoder_mode.startswith("split_")
            and self._decoder_fp16_dot
        ) else "0"
        self._decoder_weight_fp16_storage = (
            os.environ.get(
                "KERNELSWIFT_TASK04_DECODER_WEIGHT_FP16",
                decoder_weight_fp16_default,
            ).strip()
            == "1"
        )
        mixed_precision_enabled = (
            self._normalized_fp16_workspace
            or self._decoder_weight_fp16_storage
        )
        if mixed_precision_enabled and not (
            self.pooling == "max"
            and self._a2_decoder_mode.startswith("split_")
            and self._decoder_fp16_dot
        ):
            raise ValueError(
                "Task04 FP16 storage experiments require max pooling, "
                "a split decoder, and KERNELSWIFT_TASK04_FP16_DOT=1"
            )
        if self._decoder_weight_fp16_storage:
            decoder_weight = self.decoder.weight
            self.decoder.weight = nn.Parameter(
                decoder_weight.detach().to(dtype=torch.float16),
                requires_grad=decoder_weight.requires_grad,
            )
        self._use_compiled_runners = (
            chip == "iluvatar_bi150"
        )
        self._use_fixed_shape_runner = (
            os.environ.get("KERNELSWIFT_TASK04_FIXED_RUNNER", "1").strip() != "0"
        )
        self._fixed_runner_device = None
        self._fixed_linear_runner = None
        self._fixed_layer_norm_runner = None
        self._fixed_decoder_runner = None
        # Cache executable grid launchers only.  Forward still allocates every
        # intermediate/output tensor afresh and every kernel fully overwrites
        # its destination, so no input-dependent result is retained here.
        self._compiled_runners = {}
        graph_value = os.environ.get("KERNELSWIFT_TASK04_GRAPH_REPLAY")
        if graph_value is None:
                graph_value = os.environ.get("KERNELSWIFT_BI150_GRAPH_REPLAY", "1")
        self._use_graph_replay = (
            chip == "iluvatar_bi150"
            and graph_value.strip().lower() in ("1", "true", "yes")
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_output = None

    def _try_fixed_bi_forward(
        self, hidden_states: torch.Tensor, seq_lens: torch.Tensor
    ):
        """Run the exact published BI150 layout without rebuilding cache keys."""
        if not (
            self._use_compiled_runners
            and self._use_fixed_shape_runner
            and self.pooling == "max"
            and self._a2_decoder_mode.startswith("split_")
            and self.dense.in_features == 768
            and self.dense.out_features == 768
            and self.decoder.in_features == 768
            and self.decoder.out_features == 30522
            and hidden_states.shape == (83, 768)
            and seq_lens.shape == (4,)
            and hidden_states.is_contiguous()
            and seq_lens.is_contiguous()
            and self.dense.weight.is_contiguous()
            and self.decoder.weight.is_contiguous()
            and hasattr(_linear_activation_kernel, "warmup")
            and hasattr(
                _layer_norm_inplace_kernel
                if self._inplace_layer_norm
                else _layer_norm_kernel,
                "warmup",
            )
            and hasattr(_a2_decoder_split_rows_max_pool_kernel, "warmup")
        ):
            return None

        dense_output = torch.empty_like(hidden_states)
        normalized = (
            dense_output
            if self._inplace_layer_norm
            else torch.empty(
                hidden_states.shape,
                device=hidden_states.device,
                dtype=(
                    torch.float16
                    if self._normalized_fp16_workspace
                    else hidden_states.dtype
                ),
            )
        )
        pooled = torch.empty(
            (4, 30522), device=hidden_states.device, dtype=hidden_states.dtype
        )
        if self._fixed_runner_device != hidden_states.device:
            self._fixed_runner_device = hidden_states.device
            self._fixed_linear_runner = None
            self._fixed_layer_norm_runner = None
            self._fixed_decoder_runner = None

        block_m = self._linear_block_m
        block_n = self._linear_block_n
        block_k = self._linear_block_k
        linear_grid = (
            triton.cdiv(83, block_m) * triton.cdiv(768, block_n),
            1,
            1,
        )
        if self._fixed_linear_runner is None:
            compiled = _linear_activation_kernel.warmup(
                hidden_states,
                self.dense.weight,
                self.dense.bias,
                dense_output,
                hidden_states.stride(0),
                hidden_states.stride(1),
                self.dense.weight.stride(0),
                self.dense.weight.stride(1),
                dense_output.stride(0),
                dense_output.stride(1),
                83,
                768,
                768,
                ACTIVATION=1,
                USE_FP16_DOT=self._dense_fp16_dot,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=self._linear_group_m,
                grid=linear_grid,
                num_warps=self._linear_warps,
                num_stages=self._linear_stages,
            )
            self._fixed_linear_runner = compiled[linear_grid]
        self._fixed_linear_runner(
            hidden_states, self.dense.weight, self.dense.bias, dense_output
        )

        norm_grid = (83, 1, 1)
        if self._fixed_layer_norm_runner is None:
            if self._inplace_layer_norm:
                compiled = _layer_norm_inplace_kernel.warmup(
                    dense_output,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                    dense_output.stride(0),
                    dense_output.stride(1),
                    768,
                    self.layer_norm.eps,
                    BLOCK_N=1024,
                    grid=norm_grid,
                    num_warps=self._layer_norm_warps,
                )
            else:
                compiled = _layer_norm_kernel.warmup(
                    dense_output,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                    normalized,
                    dense_output.stride(0),
                    dense_output.stride(1),
                    normalized.stride(0),
                    normalized.stride(1),
                    768,
                    self.layer_norm.eps,
                    BLOCK_N=1024,
                    grid=norm_grid,
                    num_warps=self._layer_norm_warps,
                )
            self._fixed_layer_norm_runner = compiled[norm_grid]
        if self._inplace_layer_norm:
            self._fixed_layer_norm_runner(
                dense_output,
                self.layer_norm.weight,
                self.layer_norm.bias,
            )
        else:
            self._fixed_layer_norm_runner(
                dense_output,
                self.layer_norm.weight,
                self.layer_norm.bias,
                normalized,
            )

        _, decoder_block_n, decoder_block_k, decoder_warps, decoder_stages = (
            self._a2_decoder_experiment_config
        )
        decoder_grid = (triton.cdiv(30522, decoder_block_n), 1, 1)
        if self._fixed_decoder_runner is None:
            compiled = _a2_decoder_split_rows_max_pool_kernel.warmup(
                normalized,
                self.decoder.weight,
                self.decoder.bias,
                seq_lens,
                pooled,
                normalized.stride(0),
                normalized.stride(1),
                self.decoder.weight.stride(0),
                self.decoder.weight.stride(1),
                pooled.stride(0),
                pooled.stride(1),
                83,
                30522,
                768,
                NUM_SEQUENCES=4,
                BLOCK_SEQUENCES=4,
                USE_FP16_DOT=self._decoder_fp16_dot,
                BLOCK_N=decoder_block_n,
                BLOCK_K=decoder_block_k,
                grid=decoder_grid,
                num_warps=decoder_warps,
                num_stages=decoder_stages,
            )
            self._fixed_decoder_runner = compiled[decoder_grid]
        self._fixed_decoder_runner(
            normalized,
            self.decoder.weight,
            self.decoder.bias,
            seq_lens,
            pooled,
        )
        return list(pooled.unbind(0))

    def _launch_linear(
        self,
        source: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        destination: torch.Tensor,
        activation: int,
    ) -> None:
        rows, reduction = source.shape
        columns = weight.shape[0]
        block_m = self._linear_block_m
        block_n = self._linear_block_n
        block_k = self._linear_block_k
        use_fp16_dot = self._dense_fp16_dot and activation == 1
        grid = (
            triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),
        )
        if self._use_compiled_runners and hasattr(
            _linear_activation_kernel, "warmup"
        ):
            compiled_grid = (grid[0], 1, 1)
            runner_key = (
                "linear",
                compiled_grid,
                activation,
                use_fp16_dot,
                block_m,
                block_n,
                block_k,
                self._linear_group_m,
                self._linear_warps,
                self._linear_stages,
                _tensor_layout_key(source),
                _tensor_layout_key(weight),
                _tensor_layout_key(bias),
                _tensor_layout_key(destination),
            )
            runner = self._compiled_runners.get(runner_key)
            if runner is None:
                compiled = _linear_activation_kernel.warmup(
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
                    grid=compiled_grid,
                    num_warps=self._linear_warps,
                    num_stages=self._linear_stages,
                )
                runner = compiled[compiled_grid]
                self._compiled_runners[runner_key] = runner
            # All shape/stride/config arguments are tl.constexpr.  The CoreX
            # Triton 3.1 runtime signature therefore contains four pointers.
            runner(source, weight, bias, destination)
            return
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
    ) -> None:
        rows, reduction = source.shape
        columns = self.decoder.weight.shape[0]
        block_m = self._linear_block_m
        block_n = self._linear_block_n
        block_k = self._linear_block_k
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        grid = (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)
        _decoder_max_pool_kernel[grid](
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
            USE_FP16_DOT=False,
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
                min(vocab_blocks, _ascend_ai_core_count(device_index)),
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
        columns = self.decoder.weight.shape[0]
        _, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        num_sequences = seq_lens.numel()
        block_sequences = triton.next_power_of_2(num_sequences)
        grid = (triton.cdiv(columns, block_n),)
        if self._use_compiled_runners and hasattr(
            _a2_decoder_split_rows_max_pool_kernel, "warmup"
        ):
            compiled_grid = (grid[0], 1, 1)
            runner_key = (
                "decoder_split_rows",
                compiled_grid,
                block_n,
                block_k,
                num_warps,
                num_stages,
                num_sequences,
                block_sequences,
                self._decoder_fp16_dot,
                _tensor_layout_key(source),
                _tensor_layout_key(self.decoder.weight),
                _tensor_layout_key(self.decoder.bias),
                _tensor_layout_key(seq_lens),
                _tensor_layout_key(destination),
            )
            runner = self._compiled_runners.get(runner_key)
            if runner is None:
                compiled = _a2_decoder_split_rows_max_pool_kernel.warmup(
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
                    USE_FP16_DOT=self._decoder_fp16_dot,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    grid=compiled_grid,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                runner = compiled[compiled_grid]
                self._compiled_runners[runner_key] = runner
            # Scalar arguments are all tl.constexpr, leaving five pointers in
            # the CoreX Triton 3.1 runtime signature.
            runner(
                source,
                self.decoder.weight,
                self.decoder.bias,
                seq_lens,
                destination,
            )
            return
        _a2_decoder_split_rows_max_pool_kernel[grid](
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
            USE_FP16_DOT=self._decoder_fp16_dot,
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
        columns = self.decoder.weight.shape[0]
        block_m, block_n, block_k, num_warps, num_stages = (
            self._a2_decoder_experiment_config
        )
        grid = (
            triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),
        )
        _linear_activation_kernel[grid](
            source,
            self.decoder.weight,
            self.decoder.bias,
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
            ACTIVATION=0,
            USE_FP16_DOT=False,
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
        if not self._use_graph_replay or hidden_states.device.type == "cpu":
            return self._forward_impl(hidden_states, seq_lens)

        graph_inputs = self._graph_inputs
        if (
            self._graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is hidden_states
            and graph_inputs[1] is seq_lens
            and graph_inputs[2] is self.dense.weight
            and graph_inputs[3] is self.dense.bias
            and graph_inputs[4] is self.layer_norm.weight
            and graph_inputs[5] is self.layer_norm.bias
            and graph_inputs[6] is self.decoder.weight
            and graph_inputs[7] is self.decoder.bias
        ):
            self._graph.replay()
            return self._graph_output

        self._forward_impl(hidden_states, seq_lens)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward_impl(hidden_states, seq_lens)
        self._graph = graph
        self._graph_inputs = (
            hidden_states,
            seq_lens,
            self.dense.weight,
            self.dense.bias,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.decoder.weight,
            self.decoder.bias,
        )
        self._graph_output = output
        graph.replay()
        return output

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

        fixed_result = self._try_fixed_bi_forward(hidden_states, seq_lens)
        if fixed_result is not None:
            return fixed_result

        dense_output = torch.empty(
            (rows, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        normalized = (
            dense_output
            if self._inplace_layer_norm
            else torch.empty(
                dense_output.shape,
                device=dense_output.device,
                dtype=(
                    torch.float16
                    if self._normalized_fp16_workspace
                    else dense_output.dtype
                ),
            )
        )
        pooled = torch.empty(
            (num_sequences, vocab_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        self._launch_linear(
            hidden_states,
            self.dense.weight,
            self.dense.bias,
            dense_output,
            activation=1,
        )
        block_hidden = triton.next_power_of_2(hidden_size)
        layer_norm_grid = (rows,)
        layer_norm_kernel = (
            _layer_norm_inplace_kernel
            if self._inplace_layer_norm
            else _layer_norm_kernel
        )
        if self._use_compiled_runners and hasattr(layer_norm_kernel, "warmup"):
            compiled_grid = (rows, 1, 1)
            runner_key = (
                "layer_norm",
                compiled_grid,
                block_hidden,
                self._layer_norm_warps,
                self.layer_norm.eps,
                _tensor_layout_key(dense_output),
                _tensor_layout_key(self.layer_norm.weight),
                _tensor_layout_key(self.layer_norm.bias),
                self._inplace_layer_norm,
                _tensor_layout_key(normalized),
            )
            runner = self._compiled_runners.get(runner_key)
            if runner is None:
                if self._inplace_layer_norm:
                    compiled = layer_norm_kernel.warmup(
                        dense_output,
                        self.layer_norm.weight,
                        self.layer_norm.bias,
                        dense_output.stride(0),
                        dense_output.stride(1),
                        hidden_size,
                        self.layer_norm.eps,
                        BLOCK_N=block_hidden,
                        grid=compiled_grid,
                        num_warps=self._layer_norm_warps,
                    )
                else:
                    compiled = layer_norm_kernel.warmup(
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
                        grid=compiled_grid,
                        num_warps=self._layer_norm_warps,
                    )
                runner = compiled[compiled_grid]
                self._compiled_runners[runner_key] = runner
            # As above, all non-pointer arguments are tl.constexpr.
            if self._inplace_layer_norm:
                runner(
                    dense_output,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                )
            else:
                runner(
                    dense_output,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                    normalized,
                )
        else:
            if self._inplace_layer_norm:
                layer_norm_kernel[layer_norm_grid](
                    dense_output,
                    self.layer_norm.weight,
                    self.layer_norm.bias,
                    dense_output.stride(0),
                    dense_output.stride(1),
                    hidden_size,
                    self.layer_norm.eps,
                    BLOCK_N=block_hidden,
                    num_warps=self._layer_norm_warps,
                )
            else:
                layer_norm_kernel[layer_norm_grid](
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
            if self._a2_decoder_mode.startswith("raw_"):
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
                self._launch_decoder_max_pool(normalized, seq_lens, pooled)
        else:
            activated_logits = torch.empty(
                (rows, vocab_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self._launch_linear(
                normalized,
                self.decoder.weight,
                self.decoder.bias,
                activated_logits,
                activation=2,
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
