import torch
import triton
import triton.language as tl
from dlblas.utils.libentry import libentry
from dlblas.utils import register_dlblas_op, SymVar, Tensor, ChoiceSpace

@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SEQ": BS}, num_stages=s, num_warps=w)
        for BS in [1024]
        for s in [4]
        for w in [4]
    ],
    key=["max_seq_len"],
)

@triton.jit
def yarnROPE_fwd_kernel(
    inv_freq,
    max_seq_len,
    offset,
    half_head_dim: tl.constexpr,
    cos,
    sin,
    BLOCK_SEQ: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    offs_seq = seq_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    cur_seq_data = (seq_idx * BLOCK_SEQ + tl.arange(0,BLOCK_SEQ)).view(BLOCK_SEQ,1)
    if offset != 0:
        cur_seq_data += offset
    inv_freq_data = tl.load(inv_freq + tl.arange(0,half_head_dim)).view(1,half_head_dim)

    cur_emb_data = cur_seq_data * inv_freq_data

    cos_data = tl.cos(cur_emb_data) 
    sin_data = tl.sin(cur_emb_data) 

    cur_emb_data_offset0 = offs_seq[:,None] * half_head_dim * 2 + tl.arange(0,half_head_dim)[None,:]
    cur_emb_data_offset1 = cur_emb_data_offset0 + half_head_dim

    tl.store(cos + cur_emb_data_offset0, cos_data)
    tl.store(cos + cur_emb_data_offset1, cos_data)
    tl.store(sin + cur_emb_data_offset0, sin_data)
    tl.store(sin + cur_emb_data_offset1, sin_data)

class yarnROPE_withCSC(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.Any, max_seq_len, offset, inv_freq):
        half_head_dim = inv_freq.shape[0]
        cos = torch.empty(max_seq_len, half_head_dim * 2, device='mlu')
        sin = torch.empty(max_seq_len, half_head_dim * 2, device='mlu')

        assert (
            inv_freq.is_contiguous()
        )

        with torch.cuda.device(inv_freq.device):
            grid = lambda META: (
                triton.cdiv(max_seq_len, META["BLOCK_SEQ"]),
            )
            yarnROPE_fwd_kernel[grid](
                inv_freq,
                max_seq_len,
                offset,
                half_head_dim,
                cos,
                sin,
            )
        return cos, sin




