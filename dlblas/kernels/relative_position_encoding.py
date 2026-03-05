"""
RelativePositionEncoding (AF3 Algo 3-like)

From: protenix/model/modules/embedders.py:RelativePositionEncoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton kernel: fused relative position encoding + linear projection
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_I": 16, "BLOCK_J": 16, "BLOCK_C": 4}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 16, "BLOCK_J": 16, "BLOCK_C": 8}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 32, "BLOCK_J": 8, "BLOCK_C": 4}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 8, "BLOCK_J": 32, "BLOCK_C": 4}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 32, "BLOCK_J": 32, "BLOCK_C": 4}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_C": 2}, num_warps=8, num_stages=3
        ),
        # New configs with larger channel blocks and tile sizes
        triton.Config(
            {"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_C": 4}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_I": 64, "BLOCK_J": 64, "BLOCK_C": 8}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_I": 128, "BLOCK_J": 128, "BLOCK_C": 2}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_I": 32, "BLOCK_J": 32, "BLOCK_C": 16}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 16, "BLOCK_J": 16, "BLOCK_C": 16}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_I": 8, "BLOCK_J": 8, "BLOCK_C": 32}, num_warps=4, num_stages=3
        ),
    ],
    key=["N", "c_z"],
)
@triton.jit
def _relpos_kernel(
    out_ptr,
    asym_id_ptr,
    residue_ptr,
    entity_ptr,
    token_ptr,
    sym_ptr,
    emb_pos_ptr,
    emb_token_ptr,
    emb_chain_ptr,
    w_entity_ptr,
    N,
    r_max,
    s_max,
    c_z,
    pos_dim,
    token_dim,
    chain_dim,
    stride_out_i,
    stride_out_j,
    stride_out_c,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J

    # indices for the current tile
    i = i_start + tl.arange(0, BLOCK_I)
    j = j_start + tl.arange(0, BLOCK_J)

    i_mask = i < N
    j_mask = j < N

    # ----- load and broadcast input features -----
    # asym_id
    asym_i = tl.load(asym_id_ptr + i, mask=i_mask)  # (BLOCK_I,)
    asym_j = tl.load(asym_id_ptr + j, mask=j_mask)  # (BLOCK_J,)
    asym_i_exp = asym_i[:, None]  # (BLOCK_I, 1)
    asym_j_exp = asym_j[None, :]  # (1, BLOCK_J)
    same_chain = asym_i_exp == asym_j_exp  # (BLOCK_I, BLOCK_J)

    # residue_index
    ri = tl.load(residue_ptr + i, mask=i_mask)
    rj = tl.load(residue_ptr + j, mask=j_mask)
    ri_exp = ri[:, None]
    rj_exp = rj[None, :]
    same_residue = ri_exp == rj_exp

    # entity_id
    ei = tl.load(entity_ptr + i, mask=i_mask)
    ej = tl.load(entity_ptr + j, mask=j_mask)
    ei_exp = ei[:, None]
    ej_exp = ej[None, :]
    same_entity = ei_exp == ej_exp

    # token_index
    ti = tl.load(token_ptr + i, mask=i_mask)
    tj = tl.load(token_ptr + j, mask=j_mask)
    ti_exp = ti[:, None]
    tj_exp = tj[None, :]

    # sym_id
    si = tl.load(sym_ptr + i, mask=i_mask)
    sj = tl.load(sym_ptr + j, mask=j_mask)
    si_exp = si[:, None]
    sj_exp = sj[None, :]

    # ----- compute the three index matrices -----
    # residue relative index
    diff_res = ri_exp - rj_exp + r_max
    diff_res = tl.minimum(tl.maximum(diff_res, 0), 2 * r_max)
    d_res = tl.where(same_chain, diff_res, 2 * r_max + 1)  # (BLOCK_I, BLOCK_J)

    # token relative index
    diff_token = ti_exp - tj_exp + r_max
    diff_token = tl.minimum(tl.maximum(diff_token, 0), 2 * r_max)
    mask_token = same_chain & same_residue
    d_token = tl.where(mask_token, diff_token, 2 * r_max + 1)  # (BLOCK_I, BLOCK_J)

    # chain relative index (sym_id)
    diff_chain = si_exp - sj_exp + s_max
    diff_chain = tl.minimum(tl.maximum(diff_chain, 0), 2 * s_max)
    d_chain = tl.where(same_entity, diff_chain, 2 * s_max + 1)  # (BLOCK_I, BLOCK_J)

    # ----- loop over channel blocks -----
    for ch_start in range(0, c_z, BLOCK_C):
        ch = ch_start + tl.arange(0, BLOCK_C)
        ch_mask = ch < c_z

        # combined mask for this tile
        mask = i_mask[:, None, None] & j_mask[None, :, None] & ch_mask[None, None, :]

        # gather embedding contributions (load as half and convert to float32)
        # pos
        pos_offset = (
            d_res[:, :, None] * c_z + ch[None, None, :]
        )  # (BLOCK_I, BLOCK_J, BLOCK_C)
        pos_val = tl.load(emb_pos_ptr + pos_offset, mask=mask).to(tl.float32)

        # token
        token_offset = d_token[:, :, None] * c_z + ch[None, None, :]
        token_val = tl.load(emb_token_ptr + token_offset, mask=mask).to(tl.float32)

        # chain
        chain_offset = d_chain[:, :, None] * c_z + ch[None, None, :]
        chain_val = tl.load(emb_chain_ptr + chain_offset, mask=mask).to(tl.float32)

        # entity binary feature
        entity_vec = tl.load(w_entity_ptr + ch, mask=ch_mask).to(
            tl.float32
        )  # (BLOCK_C,)
        entity_vec = entity_vec[None, None, :]  # (1, 1, BLOCK_C)
        entity_contrib = tl.where(same_entity[:, :, None], entity_vec, 0.0)

        out_val = pos_val + token_val + chain_val + entity_contrib

        # compute output memory location (contiguous N,N,c_z layout)
        out_idx = (
            i[:, None, None] * stride_out_i
            + j[None, :, None] * stride_out_j
            + ch[None, None, :] * stride_out_c
        )
        tl.store(out_ptr + out_idx, out_val, mask=mask)


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
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
        N = asym_id.size(0)
        r_max = self.r_max
        s_max = self.s_max
        c_z = self.c_z

        # ---- split linear weight into embedding tables and convert to half ----
        weight = self.proj.weight  # (c_z, total_dim)
        pos_dim = 2 * r_max + 2
        token_dim = 2 * r_max + 2
        chain_dim = 2 * s_max + 2

        W_pos = weight[:, :pos_dim].t().contiguous().half()  # (pos_dim, c_z) half
        W_token = weight[:, pos_dim : pos_dim + token_dim].t().contiguous().half()
        W_chain = weight[:, pos_dim + token_dim + 1 :].t().contiguous().half()
        W_entity = (
            weight[:, pos_dim + token_dim : pos_dim + token_dim + 1]
            .squeeze(1)
            .contiguous()
            .half()
        )  # (c_z,) half

        # ---- prepare output tensor ----
        out = torch.empty(N, N, c_z, device=asym_id.device, dtype=torch.float32)

        # ---- ensure inputs are contiguous ----
        asym_id = asym_id.contiguous()
        residue_index = residue_index.contiguous()
        entity_id = entity_id.contiguous()
        token_index = token_index.contiguous()
        sym_id = sym_id.contiguous()

        # ---- launch Triton kernel ----
        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_I"]),
            triton.cdiv(N, META["BLOCK_J"]),
        )
        _relpos_kernel[grid](
            out,
            asym_id,
            residue_index,
            entity_id,
            token_index,
            sym_id,
            W_pos,
            W_token,
            W_chain,
            W_entity,
            N,
            r_max,
            s_max,
            c_z,
            pos_dim,
            token_dim,
            chain_dim,
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )
        return out


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_TOKEN = 256
N_CHAIN = 2
R_MAX = 32
S_MAX = 2
C_Z = 128


def get_inputs():
    device = "cuda"
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
