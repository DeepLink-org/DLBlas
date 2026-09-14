# Copyright (c) 2025, DeepLink.
import torch

from dlblas.kernels.engram import EngramPt, EngramTri

# EngramPt.forward reads the module-level hidden_size / hc_mult, so both modules
# are built at that shape.
HIDDEN = 1024
HC_MULT = 4


def _build(cls):
    torch.manual_seed(7)
    module = cls(HIDDEN, HIDDEN, kernel_size=1, dilation=1, hc_mult=HC_MULT, activation=False)
    with torch.no_grad():
        # exactly representable in half, so the fused path's half buffers add no
        # error of their own and the gate is the only thing under test
        module.conv.weight.fill_(1.0)
    if hasattr(module, '_precompute_buffers'):
        module._precompute_buffers()
    return module


def test_engram_gate_masked_query_rows():
    """A masked position carries an all-zero query row, which makes the gate
    argument exactly zero. torch.sign is 0 there, so the gate is sigmoid(0); the
    kernel must not gate the zero as positive."""
    engram_tri = _build(EngramTri)
    engram_pt = _build(EngramPt)

    torch.manual_seed(0)
    embeddings = torch.randn(1, 6, HIDDEN, device='cuda')
    hidden_states = torch.randn(1, 6, HC_MULT, HIDDEN, device='cuda')
    hidden_states[0, 3] = 0.0        # a padded token
    hidden_states[0, 1, 2] = 0.0     # one masked head slot

    out_tri = engram_tri(embeddings=embeddings.clone(), hidden_states=hidden_states.clone())
    out_pt = engram_pt(embeddings=embeddings.clone(), hidden_states=hidden_states.clone())

    masked = hidden_states.abs().sum(-1) == 0
    torch.testing.assert_close(out_tri[masked], out_pt[masked], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_tri, out_pt, atol=1e-3, rtol=1e-3)
