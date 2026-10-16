"""
RelativePositionEncoding (AF3 Algo 3-like)

From: protenix/model/modules/embedders.py:RelativePositionEncoding
"""

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F

# Triton accelerated fused kernel: directly computes Linear(generate_relp(...))
# without materializing the large relp tensor.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@triton.jit
def rpe_proj_kernel(
    asym_ptr, residue_ptr, entity_ptr, token_ptr, sym_ptr,
    Wt_ptr, out_ptr,
    N, C,
    r_max, s_max,
    stride_out_m, stride_out_n, stride_out_c,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    # indices along j and c
    j_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.max_contiguous(c_offsets, BLOCK_C)
    tl.max_contiguous(j_offsets, BLOCK_N)
    tl.multiple_of(c_offsets, BLOCK_C)

    mask_j = j_offsets < N
    mask_c = c_offsets < C
    pair_mask = mask_j[:, None] & mask_c[None, :]

    # i index (row). Grid ensures pid_m < N
    i = pid_m

    # Load scalar features for i and cast to int32 for lighter arithmetic
    asym_i = tl.astype(tl.load(asym_ptr + i), tl.int32)
    resid_i = tl.astype(tl.load(residue_ptr + i), tl.int32)
    entity_i = tl.astype(tl.load(entity_ptr + i), tl.int32)
    token_i = tl.astype(tl.load(token_ptr + i), tl.int32)
    sym_i = tl.astype(tl.load(sym_ptr + i), tl.int32)

    # Load vector features for j and cast to int32
    asym_j = tl.astype(tl.load(asym_ptr + j_offsets, mask=mask_j, other=0), tl.int32)
    resid_j = tl.astype(tl.load(residue_ptr + j_offsets, mask=mask_j, other=0), tl.int32)
    entity_j = tl.astype(tl.load(entity_ptr + j_offsets, mask=mask_j, other=0), tl.int32)
    token_j = tl.astype(tl.load(token_ptr + j_offsets, mask=mask_j, other=0), tl.int32)
    sym_j = tl.astype(tl.load(sym_ptr + j_offsets, mask=mask_j, other=0), tl.int32)

    # Same masks
    same_chain = asym_i == asym_j
    same_residue = resid_i == resid_j
    same_entity = entity_i == entity_j

    # Derived constants (int32)
    two_r = 2 * r_max
    P = 2 * (r_max + 1)
    sentinel_pos = two_r + 1
    two_s = 2 * s_max
    sentinel_chain = two_s + 1

    # Clamp deltas (do not map to sentinel here)
    d_res = resid_i - resid_j + r_max
    d_res = tl.maximum(d_res, 0)
    d_res = tl.minimum(d_res, two_r)

    d_tok = token_i - token_j + r_max
    d_tok = tl.maximum(d_tok, 0)
    d_tok = tl.minimum(d_tok, two_r)

    d_chain = sym_i - sym_j + s_max
    d_chain = tl.maximum(d_chain, 0)
    d_chain = tl.minimum(d_chain, two_s)

    # Offsets for concatenated one-hot segments
    offset_token = P
    offset_b = 2 * P
    offset_chain = offset_b + 1

    # Pointer math uses int64 for safety
    C_i64 = tl.full((), C, tl.int64)
    c_off_i64 = tl.astype(c_offsets, tl.int64)

    pos_row = tl.astype(d_res, tl.int64)
    tok_row = tl.astype(d_tok + offset_token, tl.int64)
    chain_row = tl.astype(d_chain + offset_chain, tl.int64)

    # Preload sentinel rows once per (j-tile, c-tile) and broadcast across j
    sent_pos_ptrs = Wt_ptr + tl.astype(sentinel_pos, tl.int64) * C_i64 + c_off_i64
    sent_tok_ptrs = Wt_ptr + tl.astype(sentinel_pos + offset_token, tl.int64) * C_i64 + c_off_i64
    sent_chain_ptrs = Wt_ptr + tl.astype(sentinel_chain + offset_chain, tl.int64) * C_i64 + c_off_i64

    pos_base = tl.load(sent_pos_ptrs, mask=mask_c, other=0.0)[None, :]   # [1, BLOCK_C]
    tok_base = tl.load(sent_tok_ptrs, mask=mask_c, other=0.0)[None, :]   # [1, BLOCK_C]
    chain_base = tl.load(sent_chain_ptrs, mask=mask_c, other=0.0)[None, :]  # [1, BLOCK_C]

    # Build pointers for valid (non-sentinel) fetches
    pos_ptrs = Wt_ptr + pos_row[:, None] * C_i64 + c_off_i64[None, :]
    tok_ptrs = Wt_ptr + tok_row[:, None] * C_i64 + c_off_i64[None, :]
    chain_ptrs = Wt_ptr + chain_row[:, None] * C_i64 + c_off_i64[None, :]

    # Masks for valid fetches; use sentinel vectors as "other" to avoid extra ops
    pos_mask = pair_mask & (same_chain[:, None])
    tok_mask = pair_mask & ((same_chain & same_residue)[:, None])
    chain_mask = pair_mask & (same_entity[:, None])

    pos_w = tl.load(pos_ptrs, mask=pos_mask, other=pos_base)
    tok_w = tl.load(tok_ptrs, mask=tok_mask, other=tok_base)
    chain_w = tl.load(chain_ptrs, mask=chain_mask, other=chain_base)

    # Single-channel b_same_entity row
    b_row_ptrs = Wt_ptr + tl.astype(offset_b, tl.int64) * C_i64 + c_off_i64
    b_row = tl.load(b_row_ptrs, mask=mask_c, other=0.0)[None, :]  # [1, BLOCK_C]
    b_factor = tl.where(same_entity, 1.0, 0.0)[:, None]  # [BLOCK_N, 1]

    # Accumulate contributions
    acc = pos_w + tok_w + chain_w + b_row * b_factor

    # Store to out [N, N, C]
    out_ptrs = out_ptr + tl.astype(i, tl.int64) * tl.astype(stride_out_m, tl.int64) \
               + tl.astype(j_offsets, tl.int64)[:, None] * tl.astype(stride_out_n, tl.int64) \
               + tl.astype(c_offsets, tl.int64)[None, :] * tl.astype(stride_out_c, tl.int64)
    tl.store(out_ptrs, acc, mask=pair_mask)


def _fused_relp_linear_triton(
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    entity_id: torch.Tensor,
    token_index: torch.Tensor,
    sym_id: torch.Tensor,
    Wt: torch.Tensor,
    r_max: int,
    s_max: int,
) -> torch.Tensor:
    """
    Triton fused path: compute proj(generate_relp(...)) without materializing relp.
    Wt is the transposed weight of the Linear layer with shape [in_dim, c_z] (contiguous).
    """
    N = asym_id.shape[0]
    C = Wt.shape[1]
    out = torch.empty((N, N, C), device=asym_id.device, dtype=torch.float32)

    # Tuned tile sizes for Ascend: larger N tile to amortize global reads, moderate C tile for register pressure
    BLOCK_N = 128
    BLOCK_C = 64
    grid = (N, triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))

    rpe_proj_kernel[grid](
        asym_id, residue_index, entity_id, token_index, sym_id,
        Wt, out,
        N, C,
        r_max, s_max,
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    return out


def generate_relp(
    *,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    entity_id: torch.Tensor,
    token_index: torch.Tensor,
    sym_id: torch.Tensor,
    r_max: int = 32,
    s_max: int = 2,
) -> torch.Tensor:
    """
    生成 relp 特征（和仓库实现一致的特征集合与 one-hot 形状）：
    relp = concat([a_rel_pos, a_rel_token, b_same_entity, a_rel_chain])

    Inputs: 都是 [N_token] int64
    Output: [N_token, N_token, (4*r_max + 2*s_max + 7)]
    """
    # same_* masks: [N,N]
    b_same_chain = (asym_id[:, None] == asym_id[None, :]).long()
    b_same_residue = (residue_index[:, None] == residue_index[None, :]).long()
    b_same_entity = (entity_id[:, None] == entity_id[None, :]).long()

    # residue relative index one-hot: size = 2*(r_max+1)
    d_residue = torch.clamp(
        residue_index[:, None] - residue_index[None, :] + r_max, min=0, max=2 * r_max
    )
    d_residue = d_residue * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
    a_rel_pos = F.one_hot(d_residue, 2 * (r_max + 1))

    # token relative index one-hot: size = 2*(r_max+1)
    d_token = torch.clamp(
        token_index[:, None] - token_index[None, :] + r_max, min=0, max=2 * r_max
    )
    d_token = d_token * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
        2 * r_max + 1
    )
    a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))

    # sym_id relative one-hot: size = 2*(s_max+1)
    d_chain = torch.clamp(sym_id[:, None] - sym_id[None, :] + s_max, min=0, max=2 * s_max)
    d_chain = d_chain * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
    a_rel_chain = F.one_hot(d_chain, 2 * (s_max + 1))

    relp = torch.cat(
        [
            a_rel_pos,
            a_rel_token,
            b_same_entity[..., None],
            a_rel_chain,
        ],
        dim=-1,
    ).float()
    return relp


class ModelNew(nn.Module):
    """
    相对位置编码：raw features -> relp -> pair embedding (线性投影到 c_z)
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        in_dim = 4 * r_max + 2 * s_max + 7
        self.proj = nn.Linear(in_dim, c_z, bias=False)

    def forward(
        self,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        entity_id: torch.Tensor,
        token_index: torch.Tensor,
        sym_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            asym_id: [N_token]
            residue_index: [N_token]
            entity_id: [N_token]
            token_index: [N_token]
            sym_id: [N_token]
        Returns:
            z_rel: [N_token, N_token, c_z]
        """
        use_triton = (
            _TRITON_AVAILABLE
            and asym_id.is_cuda is False  # avoid CUDA misrouting
            and asym_id.device.type == 'npu'
            and self.proj.weight.dtype == torch.float32
        )
        if use_triton:
            # Use fused Triton kernel: Linear(generate_relp(...))
            # Prepare weight as [in_dim, c_z]
            Wt = self.proj.weight.t().contiguous()
            return _fused_relp_linear_triton(
                asym_id=asym_id,
                residue_index=residue_index,
                entity_id=entity_id,
                token_index=token_index,
                sym_id=sym_id,
                Wt=Wt,
                r_max=self.r_max,
                s_max=self.s_max,
            )
        # Fallback to reference path
        relp = generate_relp(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            r_max=self.r_max,
            s_max=self.s_max,
        )
        return self.proj(relp)


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_TOKEN = 256
N_CHAIN = 2
R_MAX = 32
S_MAX = 2
C_Z = 128


def get_inputs():
    device = 'npu'
    torch.manual_seed(42)

    # Generate minimal but "semantically correct" toy input
    asym_id = torch.arange(N_TOKEN, device=device) % max(1, N_CHAIN)
    residue_index = torch.arange(N_TOKEN, device=device)
    entity_id = asym_id.clone()
    token_index = torch.arange(N_TOKEN, device=device)
    sym_id = torch.zeros(N_TOKEN, device=device, dtype=torch.long)

    # Return raw features as a list (not pre-computed relp)
    return [asym_id, residue_index, entity_id, token_index, sym_id]


def get_init_inputs():
    return [R_MAX, S_MAX, C_Z]
