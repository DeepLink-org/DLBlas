# Copyright (c) 2025, DeepLink.
import pytest
import torch

from dlblas.kernels.permute.moe_triton_kernels import moe_permute_topk_bwd_op
from dlblas.utils.device_utils import infer_device

device = infer_device()


def _reference_rows(go, pos, tokens, num_topK):
    # same accumulation order and dtype as permute_bwd_kernel: bf16, k-sequential
    out = []
    for t in tokens:
        acc = torch.zeros(go.shape[1], device=go.device, dtype=go.dtype)
        for k in range(num_topK):
            acc = acc + go[pos[t, k]]
        out.append(acc)
    return torch.stack(out)


def _position_table(sorted_row_id, num_tokens, num_topK):
    # pos[t, k] = permuted row that holds slot (t, k)
    num_elements = num_tokens * num_topK
    pos = torch.empty(num_elements, dtype=torch.long, device=sorted_row_id.device)
    pos[sorted_row_id] = torch.arange(num_elements, device=sorted_row_id.device)
    return pos.view(num_tokens, num_topK)


@pytest.mark.parametrize("num_tokens, num_topK, hidden", [
    (128, 8, 512),
    (1000, 4, 1000),
])
def test_moe_permute_topk_bwd(num_tokens, num_topK, hidden):
    torch.manual_seed(0)
    num_elements = num_tokens * num_topK
    go = torch.randn(num_elements, hidden, device=device, dtype=torch.bfloat16)
    sorted_row_id = torch.randperm(num_elements, device=device)

    act_grad = moe_permute_topk_bwd_op(go, sorted_row_id, (num_tokens, hidden), num_topK)

    pos = _position_table(sorted_row_id, num_tokens, num_topK)
    ref = _reference_rows(go, pos, range(num_tokens), num_topK)
    assert torch.equal(act_grad, ref)


@pytest.mark.skipif(
    torch.cuda.get_device_properties(0).total_memory < 10 * 2**30,
    reason="needs a >2**31-element gradient buffer (~5 GB)",
)
def test_moe_permute_topk_bwd_large_offsets():
    # row offsets must be computed in 64-bit: perm_row * num_cols exceeds
    # 2**31 once the permuted gradient buffer passes 2**31 elements
    # (hidden=7168, topK=8 reaches that at ~37.4k tokens)
    torch.manual_seed(0)
    num_tokens, num_topK, hidden = 37600, 8, 7168
    num_elements = num_tokens * num_topK
    assert num_elements * hidden > 2**31
    wrap_row = 2**31 // hidden

    go = torch.randn(num_elements, hidden, device=device, dtype=torch.bfloat16)
    sorted_row_id = torch.randperm(num_elements, device=device)

    act_grad = moe_permute_topk_bwd_op(go, sorted_row_id, (num_tokens, hidden), num_topK)
    torch.cuda.synchronize()

    # spot-check tokens that gather rows past the wrap threshold, plus controls
    pos = _position_table(sorted_row_id, num_tokens, num_topK)
    above = (pos >= wrap_row).any(dim=1).nonzero().flatten()
    sample = above[:16].tolist() + [0, 1, num_tokens // 2]
    ref = _reference_rows(go, pos, sample, num_topK)
    assert torch.equal(act_grad[sample], ref)
