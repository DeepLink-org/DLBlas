"""
RelativePositionEncoding (AF3 Algo 3-like)

From: protenix/model/modules/embedders.py:RelativePositionEncoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


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


@triton.jit
def relp_proj_kernel(
    asym_id_ptr, residue_index_ptr, entity_id_ptr, token_index_ptr, sym_id_ptr,
    W_ptr, out_ptr,
    N, C, r_max, s_max, in_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_i = offs_i < N
    mask_j = offs_j < N

    # Load 1D features for i and j blocks (int32 for cheaper ops)
    ai = tl.load(asym_id_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)
    aj = tl.load(asym_id_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)
    ri = tl.load(residue_index_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)
    rj = tl.load(residue_index_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)
    ei = tl.load(entity_id_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)
    ej = tl.load(entity_id_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)
    ti = tl.load(token_index_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)
    tj = tl.load(token_index_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)
    si = tl.load(sym_id_ptr + offs_i, mask=mask_i, other=0).to(tl.int32)
    sj = tl.load(sym_id_ptr + offs_j, mask=mask_j, other=0).to(tl.int32)

    # Broadcast to tile
    ai_mat = ai[:, None]
    aj_mat = aj[None, :]
    ri_mat = ri[:, None]
    rj_mat = rj[None, :]
    ei_mat = ei[:, None]
    ej_mat = ej[None, :]
    ti_mat = ti[:, None]
    tj_mat = tj[None, :]
    si_mat = si[:, None]
    sj_mat = sj[None, :]

    # Same-* masks
    b_same_chain = (ai_mat == aj_mat).to(tl.int32)
    b_same_residue = (ri_mat == rj_mat).to(tl.int32)
    b_same_entity = (ei_mat == ej_mat).to(tl.int32)
    bse_f = b_same_entity.to(tl.float32)

    # Precompute constants
    two_r = 2 * r_max
    two_rp1 = two_r + 1
    two_s = 2 * s_max
    two_sp1 = two_s + 1

    # Residue relative indices with bucket for different chain
    d_residue = ri_mat - rj_mat + r_max
    d_residue = tl.maximum(d_residue, 0)
    d_residue = tl.minimum(d_residue, two_r)
    d_residue = d_residue * b_same_chain + (1 - b_same_chain) * two_rp1

    # Token relative indices with bucket for different chain*residue
    d_token = ti_mat - tj_mat + r_max
    d_token = tl.maximum(d_token, 0)
    d_token = tl.minimum(d_token, two_r)
    bc = b_same_chain * b_same_residue
    d_token = d_token * bc + (1 - bc) * two_rp1

    # Chain (sym_id) relative indices with bucket for different entity
    d_chain = si_mat - sj_mat + s_max
    d_chain = tl.maximum(d_chain, 0)
    d_chain = tl.minimum(d_chain, two_s)
    d_chain = d_chain * b_same_entity + (1 - b_same_entity) * two_sp1

    # One-hot segment lengths/offsets
    Lpos = 2 * (r_max + 1)      # 2*r_max + 2
    Ltok = Lpos                 # 2*r_max + 2
    off_tok = Lpos
    off_be = Lpos + Ltok
    off_chain = off_be + 1

    mask_ij = mask_i[:, None] & mask_j[None, :]

    # Base output pointer for this (i,j) tile
    out_base = out_ptr + (offs_i[:, None] * (N * C) + offs_j[None, :] * C)

    # Vectorize over the BLOCK_C channels in one shot (instead of per-kk loop)
    offs_k = tl.arange(0, BLOCK_C)
    tl.multiple_of(offs_k, BLOCK_C)
    k_idx = pid_c * BLOCK_C + offs_k
    k_mask = k_idx < C

    # Row base pointer per channel k (broadcast to [BM, BN, KC])
    row_base_k = W_ptr + (k_idx[None, None, :] * in_dim)

    # Build pointers for the three one-hot segments, shape: [BM, BN, KC]
    pos_ptrs = row_base_k + d_residue[:, :, None]
    tok_ptrs = row_base_k + (off_tok + d_token)[:, :, None]
    chain_ptrs = row_base_k + (off_chain + d_chain)[:, :, None]

    mask_ijk = mask_ij[:, :, None] & k_mask[None, None, :]

    # Load contributions; cache in L1 to reuse across pos/tok/chain accesses
    pos_vals = tl.load(pos_ptrs, mask=mask_ijk, other=0.0, cache_modifier=".ca")
    tok_vals = tl.load(tok_ptrs, mask=mask_ijk, other=0.0, cache_modifier=".ca")
    chain_vals = tl.load(chain_ptrs, mask=mask_ijk, other=0.0, cache_modifier=".ca")

    # Load b_same_entity column weight per channel and broadcast
    w_be_ptrs = W_ptr + k_idx * in_dim + off_be
    w_be = tl.load(w_be_ptrs, mask=k_mask, other=0.0, cache_modifier=".ca")  # [KC]

    out_vals = pos_vals + tok_vals + chain_vals + w_be[None, None, :] * bse_f[:, :, None]

    # Store into [N, N, C] with last dim contiguous; pointer shape [BM, BN, KC]
    out_ptrs = out_base[:, :, None] + k_idx[None, None, :]
    tl.store(out_ptrs, out_vals, mask=mask_ijk)


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
        # Triton kernel directly computes the projection without materializing the one-hot tensor.
        N = asym_id.shape[0]
        C = self.c_z
        in_dim = self.proj.in_features
        # Output tensor
        z = torch.empty((N, N, C), device=asym_id.device, dtype=self.proj.weight.dtype)

        # Ensure inputs and weights are contiguous
        asym_id_c = asym_id.contiguous()
        residue_index_c = residue_index.contiguous()
        entity_id_c = entity_id.contiguous()
        token_index_c = token_index.contiguous()
        sym_id_c = sym_id.contiguous()
        W = self.proj.weight.contiguous()

        # Launch kernel
        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_C = 16
        grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C))
        relp_proj_kernel[grid](
            asym_id_c, residue_index_c, entity_id_c, token_index_c, sym_id_c,
            W, z,
            N, C, self.r_max, self.s_max, in_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        return z


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_TOKEN = 256
N_CHAIN = 2
R_MAX = 32
S_MAX = 2
C_Z = 128


def get_inputs():
    device = 'cuda'
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