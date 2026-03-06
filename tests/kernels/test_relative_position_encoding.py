"""
RelativePositionEncoding (AF3 Algo 3-like)

From: protenix/model/modules/embedders.py:RelativePositionEncoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    d_token = d_token * b_same_chain * b_same_residue + (
        1 - b_same_chain * b_same_residue
    ) * (2 * r_max + 1)
    a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))

    # sym_id relative one-hot: size = 2*(s_max+1)
    d_chain = torch.clamp(
        sym_id[:, None] - sym_id[None, :] + s_max, min=0, max=2 * s_max
    )
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


class Model(nn.Module):
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
        # Generate relp features
        relp = generate_relp(
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            r_max=self.r_max,
            s_max=self.s_max,
        )
        # Project to c_z dimensions
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
