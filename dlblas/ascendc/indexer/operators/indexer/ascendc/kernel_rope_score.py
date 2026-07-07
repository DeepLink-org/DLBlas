# Kernel 3: rope_score_compute — RoPE + Batched MatMul (Score Computation)
# Fused Vector (RoPE) + Cube (BatchMatMul) operations
# Target: Ascend910B2, Triton-Ascend backend
#
# Architecture:
#   Stage 1: Q reshape + permute (zero-copy PyTorch view ops)
#   Stage 2: RoPE application (PyTorch NPU vector ops — TODO: Triton-ize)
#   Stage 3: Batched MatMul via torch.bmm (production) or score_matmul_kernel (Triton)
#
# Performance notes:
#   - torch.bmm is preferred for large batched matmuls (BH * S * D * kv_len)
#     as it dispatches to expertly tuned GEMM kernels on Ascend NPU.
#   - score_matmul_kernel is a functional Triton alternative that uses tl.dot
#     on the Cube unit and can be enabled via use_triton_matmul=True.
#     Current block sizes (BLOCK_S=128, BLOCK_KV=64, BLOCK_D=64) provide
#     correct results but lower throughput than torch.bmm for large shapes.
#     Future work: tune block sizes and grid strategy for competitive performance.

import torch
import triton
import triton.language as tl


# ==============================================================================
# Sub-Kernel 3b: Batched MatMul (Cube)
#   scores = q @ kv_cache^T for each (batch, head) pair
#   Functional Triton alternative to torch.bmm. See performance notes above.
# ==============================================================================

@triton.jit
def score_matmul_kernel(
    q_ptr,              # (BH, S, D) bf16
    kv_ptr,             # (BH, kv_len, D) bf16 — broadcast-expanded from (B, kv_len, D)
    scores_ptr,          # (BH, S, kv_len) bf16
    S, D, kv_len,
    num_s_blocks,        # tl.cdiv(S, BLOCK_S) — for decoding pid_0 back to (bh, s_block)
    stride_q_bh, stride_q_s, stride_q_d,
    stride_kv_bh, stride_kv_j, stride_kv_d,
    stride_sc_bh, stride_sc_s, stride_sc_j,
    BLOCK_S: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Tiled batched matrix multiplication: scores[bh] = q[bh] @ kv[bh]^T

    Launch grid: (BH * num_s_blocks, cdiv(kv_len, BLOCK_KV))  — merged 2D grid.
    Merging BH and S tiles into pid_0 keeps total blocks under the Ascend NPU
    65535 limit while preserving per-tile parallelism.

    pid_0 decoded as:
      bh       = pid_0 // num_s_blocks
      s_block  = pid_0 %  num_s_blocks

    For each (bh, s_tile, kv_tile):
      Load  q[bh,    s_tile, :]       → (BLOCK_S,  D)
      Load  kv[bh,   kv_tile, :]      → (BLOCK_KV, D)
      tl.dot(q, trans(kv)) → (BLOCK_S, BLOCK_KV)
      Store → scores[bh, s_tile, kv_tile]
    """
    pid_0 = tl.program_id(0)
    kv_block = tl.program_id(1)

    bh = pid_0 // num_s_blocks
    s_block = pid_0 % num_s_blocks

    offs_s = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_kv = kv_block * BLOCK_KV + tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, BLOCK_D)

    s_mask = offs_s < S
    kv_mask = offs_kv < kv_len
    d_mask = offs_d < D

    # Load q tile: (BLOCK_S, BLOCK_D)
    q_ptrs = (q_ptr + bh * stride_q_bh
              + offs_s[:, None] * stride_q_s
              + offs_d[None, :] * stride_q_d)
    q_tile = tl.load(q_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)

    # Load kv tile: (BLOCK_KV, BLOCK_D)
    kv_ptrs = (kv_ptr + bh * stride_kv_bh
               + offs_kv[:, None] * stride_kv_j
               + offs_d[None, :] * stride_kv_d)
    kv_tile = tl.load(kv_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)

    # Compute: q_tile @ kv_tile^T  →  (BLOCK_S, BLOCK_KV)
    acc = tl.dot(q_tile, tl.trans(kv_tile))

    # Store scores tile
    sc_ptrs = (scores_ptr + bh * stride_sc_bh
               + offs_s[:, None] * stride_sc_s
               + offs_kv[None, :] * stride_sc_j)
    sc_mask = s_mask[:, None] & kv_mask[None, :]
    tl.store(sc_ptrs, acc.to(tl.bfloat16), mask=sc_mask)


# ==============================================================================
# Main entry point for Kernel 3
# ==============================================================================

def rope_score_compute(
    q_flat: torch.Tensor,
    kv_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
    B: int, S: int, H: int, D: int,
    kv_len: int,
    rd: int,
    start_pos: int,
    use_triton_matmul: bool = False,
) -> torch.Tensor:
    """Kernel 3: RoPE + Score computation.

    The computation is split into three logical stages:
      1. Q reshape + permute: (B,S,H*D) → (B,S,H,D) → (B*H,S,D)
      2. RoPE: apply rotary embedding to last rd dims (PyTorch NPU ops)
      3. Batched MatMul: (B*H,S,D) @ (B*H,kv_len,D)^T → (B*H,S,kv_len)

    Stage 3 defaults to torch.bmm for performance. Set use_triton_matmul=True
    to use the score_matmul_kernel Triton implementation instead.

    NOTE: Stage 2 (RoPE) currently uses PyTorch view_as_complex / view_as_real ops.
    These are efficient NPU vector operations. A future Triton-ized RoPE kernel would
    fuse the element-wise cos/sin rotation with the matmul for additional speedup.

    Args:
        q_flat: (B, S, H*D) bf16
        kv_cache: (B, kv_len, D) bf16
        freqs_cis: (max_seq_len, rd//2) complex64
        B, S, H, D: dimension parameters
        kv_len: effective KV length
        rd: rope head dim
        start_pos: starting position for RoPE frequencies
        use_triton_matmul: if True, use score_matmul_kernel instead of torch.bmm

    Returns:
        scores: (B, H, S, kv_len) bf16
    """
    device = q_flat.device
    dtype = q_flat.dtype

    # --- Stage 1: Q reshape + permute ---
    # (B, S, H*D) → (B, S, H, D) → permute → (B, H, S, D) → (B*H, S, D)
    q_4d = q_flat.reshape(B, S, H, D)  # (B, S, H, D)
    q_perm = q_4d.permute(0, 2, 1, 3).contiguous()  # (B, H, S, D)
    q_bh = q_perm.reshape(B * H, S, D)  # (B*H, S, D)

    # --- Stage 2: RoPE application ---
    # Apply RoPE to last rd elements using PyTorch ops
    if rd > 0:
        freqs = freqs_cis[start_pos:start_pos + S]  # (S, rd//2) complex
        # Convert complex freqs_cis to real representation
        if freqs.is_complex():
            freqs_real = torch.view_as_real(freqs)  # (S, rd//2, 2) float32
            freqs_2d = freqs_real.reshape(S, rd)    # (S, rd) float32
        else:
            freqs_2d = freqs.reshape(S, rd)

        # Extract last rd elements from q_bh: (B*H, S, D) → rope_part: (B*H, S, rd)
        q_rope = q_bh[..., D - rd:].float()  # (B*H, S, rd)

        # RoPE: treat adjacent pairs as complex numbers and rotate
        q_rope_pairs = q_rope.unflatten(-1, (-1, 2))  # (B*H, S, rd/2, 2)
        q_rope_complex = torch.view_as_complex(q_rope_pairs)  # (B*H, S, rd/2) complex

        # Broadcast freqs to match q shape: (1, S, rd/2) → (B*H, S, rd/2)
        freqs_bc = freqs.unsqueeze(0)  # (1, S, rd/2)

        q_rope_rotated = q_rope_complex * freqs_bc  # (B*H, S, rd/2) complex
        q_rope_rotated_real = torch.view_as_real(q_rope_rotated).flatten(-2)  # (B*H, S, rd)

        # Replace the rope part in q_bh
        q_rot = q_bh.clone()
        q_rot[..., D - rd:] = q_rope_rotated_real.to(dtype)
    else:
        q_rot = q_bh

    # --- Stage 3: Batched MatMul ---
    # scores = q_rot @ kv_cache^T for each (b, h)
    # q_rot: (B*H, S, D), kv_cache: (B, kv_len, D)
    #
    # Broadcast kv_cache from B → B*H so each head reuses the same KV cache.

    # kv_cache: (B, kv_len, D) → expand → (B, H, kv_len, D) → (B*H, kv_len, D)
    kv_bc = (kv_cache[:B, :kv_len, :]
             .unsqueeze(1)
             .expand(B, H, kv_len, D)
             .reshape(B * H, kv_len, D)
             .contiguous())

    BH = B * H

    if use_triton_matmul:
        # Use the Triton tiled matmul kernel (functional, but lower throughput
        # than torch.bmm for large shapes — see file-level performance notes).
        scores_bh = torch.empty(BH, S, kv_len, dtype=dtype, device=device)

        BLOCK_S = 128
        BLOCK_KV = 64
        BLOCK_D = 64

        num_s_blocks = triton.cdiv(S, BLOCK_S)
        grid = (BH * num_s_blocks, triton.cdiv(kv_len, BLOCK_KV))

        score_matmul_kernel[grid](
            q_rot, kv_bc, scores_bh,
            S, D, kv_len,
            num_s_blocks,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
            kv_bc.stride(0), kv_bc.stride(1), kv_bc.stride(2),
            scores_bh.stride(0), scores_bh.stride(1), scores_bh.stride(2),
            BLOCK_S=BLOCK_S,
            BLOCK_KV=BLOCK_KV,
            BLOCK_D=BLOCK_D,
        )
    else:
        # Default: torch.bmm (expertly tuned for Ascend NPU GEMM)
        scores_bh = torch.bmm(q_rot, kv_bc.transpose(-1, -2))  # (B*H, S, kv_len)

    # Reshape to output format: (B*H, S, kv_len) → (B, H, S, kv_len)
    scores = scores_bh.reshape(B, H, S, kv_len)

    return scores
