"""Low-precision varlen column loads: alignment, zero fill, and cache replay."""

import cutlass
import pytest
import torch

from cute_gemm.epilogue import gemm_epilogue
from cute_gemm.epilogue.ops import ColVecLoad, RowVecReduce
from cute_gemm.gemm_config import GemmConfig, TuneMode


@gemm_epilogue(ops={"scores": ColVecLoad("scores")}, reduces={"total": RowVecReduce("total")})
def _add_scores(acc, scores):
    return {"D": acc + scores, "total": acc + scores}


@gemm_epilogue(ops={"scores": ColVecLoad("scores")})
def _scale_scores(acc, scores):
    return {"D": acc * scores}


@gemm_epilogue()
def _inferred_scores(acc, scores):
    return {"D": acc * scores}


def _inputs(counts):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires SM90")
    r = sum(counts)
    a = torch.full((r, 128), 0.125, device="cuda", dtype=torch.bfloat16)
    b = torch.full((len(counts), 128, 64), 0.25, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda").int()
    return a, b, offsets


_CONFIG = GemmConfig(tile_m=128, tile_n=128, cluster_m=1, cluster_n=1, pingpong=False)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_lowbit_colvec_windows_and_semantic_zero_fill(dtype):
    # Single-row experts cover every byte residue; the last crosses tile_M.
    counts = [1] * 16 + [0, 129]
    a, b, offsets = _inputs(counts)
    scores = ((torch.arange(sum(counts), device="cuda") % 15 - 7).float() / 16).to(dtype)
    result = _add_scores(
        a, b, scores=scores, cu_seqlens_m=offsets, config=_CONFIG, out_dtype=torch.float32
    )
    expected = (4 + scores.float())[:, None].expand(-1, 64)
    torch.testing.assert_close(result["D"], expected, atol=0, rtol=0)
    totals = torch.stack(
        [expected[int(offsets[j]) : int(offsets[j + 1])].sum(0) for j in range(len(counts))]
    )
    torch.testing.assert_close(result["total"], totals, atol=0, rtol=0)


@pytest.mark.parametrize("mod", [_scale_scores, _inferred_scores])
@pytest.mark.parametrize("bad_layout", ["stride", "alignment", "length"])
def test_lowbit_colvec_rejects_before_autotune(monkeypatch, mod, bad_layout):
    from cute_gemm.gemm_runtime import autotune

    a, b, offsets = _inputs([65, 0, 194])
    r = a.shape[0]
    scores = torch.ones(2 * r, device="cuda", dtype=torch.bfloat16)
    if bad_layout == "stride":
        scores = scores[::2]
    elif bad_layout == "alignment":
        scores = scores[1 : r + 1]
    else:
        scores = scores[: r - 1]

    def unexpected_tuner(*args, **kwargs):
        pytest.fail("invalid column input reached the autotune candidate loop")

    monkeypatch.setattr(autotune, "_get_tuner", unexpected_tuner)
    with pytest.raises(ValueError, match="stride=1|16-byte|elements"):
        autotune.tuned_mod_gemm(
            mod,
            a,
            b,
            torch.empty((r, 64), device="cuda", dtype=a.dtype),
            epi_args={"scores": scores},
            b_kn=True,
            cu_seqlens_m=offsets,
            tune_mode=TuneMode.STATIC,
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_lowbit_colvec_checks_replayed_pointer(compiled):
    a, b, offsets = _inputs([65, 0, 194])
    scores = torch.ones(a.shape[0], device="cuda", dtype=torch.bfloat16)

    def call(scores):
        return _scale_scores(a, b, scores=scores, cu_seqlens_m=offsets, config=_CONFIG)["D"]

    # Inductor clones misaligned graph inputs before custom ops. aot_eager
    # preserves the caller's pointer and tests this op's runtime guard itself.
    fn = torch.compile(call, fullgraph=True, backend="aot_eager") if compiled else call
    torch.testing.assert_close(
        fn(scores), torch.full((a.shape[0], 64), 4.0, device="cuda", dtype=a.dtype), atol=0, rtol=0
    )
    # Same shape/dtype/stride, new address: metadata cache keys cannot reject this.
    bad = torch.ones(a.shape[0] + 1, device="cuda", dtype=scores.dtype)[1:]
    with pytest.raises(ValueError, match="16-byte"):
        fn(bad)


def test_fp32_colvec_keeps_identity_converter():
    a, b, offsets = _inputs([65, 0, 194])
    scores = torch.ones(a.shape[0], device="cuda")
    result = _scale_scores(a, b, scores=scores, cu_seqlens_m=offsets, config=_CONFIG)
    torch.testing.assert_close(result["D"], torch.full_like(result["D"], 4), atol=0, rtol=0)
    plans = list(_scale_scores._plan_cache.values())
    assert any(
        name == "scores" and key == (cutlass.Float32, 1) and convert is None
        for plan in plans
        for name, convert, key in plan.call_ops
    )
