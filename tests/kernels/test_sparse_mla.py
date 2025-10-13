import pytest
import torch
from dlblas.kernels.sparse_mla import sparse_mla_fwd_interface
from dlblas.utils.utils import assert_tensors_similar

def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device=q.device).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("S", [4096])
@pytest.mark.parametrize("SKV", [4096, 8192])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("HKV", [1])
@pytest.mark.parametrize("DQK", [576])
@pytest.mark.parametrize("DV", [512])
@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_sparse_mla_fwd(B, S, SKV, H, HKV, DQK, DV, topk, dtype, device):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device=device).requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    
    triton_out, triton_lse = sparse_mla_fwd_interface(q.clone(), kv.clone(), indices.clone())
    ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
    assert_tensors_similar(triton_out, ref_out, eps=1e-2, name="out")
    print("assert_tensors_similar passed")
    