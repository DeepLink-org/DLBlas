# ============================================================================
# Golden computation for big_fuse operator
# Uses the reference PyTorch implementation from origin/big_fuse.py
# ============================================================================

import torch
import os
import sys

# Constants
N_TOKENS = 512
MHC_MULT = 4
HIDDEN_SIZE = 1280
RGS = MHC_MULT * HIDDEN_SIZE  # 5120
MHC_MULT3 = 2 * MHC_MULT + MHC_MULT * MHC_MULT  # 24

RMS_EPS = 1e-6
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT_VALUE = 1.0
SINKHORN_REPEAT = 10


def _mhc_pre_norm_fn(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    """RMS-normalized linear projection (S1 in big_fuse pipeline)."""
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    n0, n1 = residual.shape[:2]
    x = residual.flatten(2, 3).float().reshape(n0 * n1, -1)
    mixes = x @ mhc_fn.T
    sqrsum = x.square().sum(-1, keepdim=True)
    mixes = mixes * (sqrsum / x.shape[-1] + mhc_norm_eps).rsqrt()
    return mixes.view(n0, n1, -1)


def _mhc_pre_split_mixes(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split mixing logits (S2 in big_fuse pipeline)."""
    a, b = input_mixes.shape[:2]
    scale = torch.cat([
        mhc_scale[0].expand(mhc_mult),
        mhc_scale[1].expand(mhc_mult),
        mhc_scale[2].expand(mhc_mult * mhc_mult),
    ])
    input_mixes = input_mixes * scale + mhc_base
    pre_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_mix = (input_mixes[:, :, mhc_mult:2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_mix = input_mixes[:, :, 2 * mhc_mult:].view(a, b, mhc_mult, mhc_mult)
    return pre_mix, post_mix, comb_mix


def _sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn doubly-stochastic normalization (S3 in big_fuse pipeline)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def _mhc_pre_apply_mix(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Weighted mix (S4 in big_fuse pipeline)."""
    return (x * mix).sum(-2).bfloat16()


def compute_golden(residual, fn, mhc_scale, mhc_base):
    """
    Compute the full big_fuse pipeline.

    Args:
        residual:   [1, 512, 4, 1280] bf16
        fn:         [24, 5120] fp32
        mhc_scale:  [3] fp32
        mhc_base:   [24] fp32

    Returns:
        post_mix:   [1, 512, 4, 1] fp32
        comb_mix:   [1, 512, 4, 4] fp32
        layer_input: [1, 512, 1280] bf16
    """
    # S1: RMS-normalized linear projection
    mixes = _mhc_pre_norm_fn(residual, fn, None, RMS_EPS)

    # S2: Split mixing logits
    pre_mix, post_mix, comb_mix = _mhc_pre_split_mixes(
        mixes, mhc_scale, mhc_base,
        MHC_MULT, MHC_POST_MULT_VALUE, MHC_PRE_EPS,
    )

    # S3: Sinkhorn normalization
    comb_mix = _sinkhorn_normalize(comb_mix, repeat=SINKHORN_REPEAT, eps=MHC_SINKHORN_EPS)

    # S4: Weighted apply
    layer_input = _mhc_pre_apply_mix(residual, pre_mix)

    return post_mix, comb_mix, layer_input


if __name__ == "__main__":
    # Read inputs from binary files
    import numpy as np

    os.makedirs("output", exist_ok=True)

    # Correct bf16 loading: data stored as uint16 (bf16 bits: upper 16 bits of fp32)
    # Pad with 16 zero bits → reinterpret as fp32 → convert to PyTorch bfloat16
    residual_u16 = np.fromfile("input/residual.bin", dtype=np.uint16).reshape(1, N_TOKENS, MHC_MULT, HIDDEN_SIZE)
    residual_u32 = residual_u16.astype(np.uint32) << 16
    residual_f32_corrected = residual_u32.view(np.float32)
    residual_t = torch.from_numpy(residual_f32_corrected).bfloat16()

    fn = np.fromfile("input/fn.bin", dtype=np.float32)
    fn_t = torch.from_numpy(fn.reshape(MHC_MULT3, RGS))

    mhc_scale = np.fromfile("input/mhc_scale.bin", dtype=np.float32)
    mhc_scale_t = torch.from_numpy(mhc_scale)

    mhc_base = np.fromfile("input/mhc_base.bin", dtype=np.float32)
    mhc_base_t = torch.from_numpy(mhc_base)

    # Compute golden
    post_mix, comb_mix, layer_input = compute_golden(
        residual_t, fn_t, mhc_scale_t, mhc_base_t
    )

    # Save outputs (flattened for C++ compatibility)
    # AscendC produces: post_mix [512, 4], comb_mix [512, 4, 4], layer_input [512, 1280]
    post_mix_out = post_mix.squeeze(-1).reshape(N_TOKENS, MHC_MULT).float().cpu().numpy()
    comb_mix_out = comb_mix.reshape(N_TOKENS, MHC_MULT, MHC_MULT).float().cpu().numpy()

    # layer_input: convert bf16 to fp32 for numpy storage, then truncate to bf16 (uint16)
    layer_input_f32 = layer_input.reshape(N_TOKENS, HIDDEN_SIZE).float().cpu().numpy()
    # Convert fp32 to bf16 uint16 representation
    layer_input_u32 = layer_input_f32.view(np.uint32)
    layer_input_u16 = (layer_input_u32 >> 16).astype(np.uint16)

    post_mix_out.tofile("output/post_mix_golden.bin")
    comb_mix_out.tofile("output/comb_mix_golden.bin")
    layer_input_u16.tofile("output/layer_input_golden.bin")

    print(f"Golden computed and saved:")
    print(f"  post_mix:    {post_mix_out.shape}, dtype=float32")
    print(f"  comb_mix:    {comb_mix_out.shape}, dtype=float32")
    print(f"  layer_input: {layer_input_f32.shape}, dtype=bfloat16 (stored as uint16)")
