"""Tests for the generic @gemm_epilogue mod autotuner (cute_gemm.gemm_runtime.autotune).

Small shapes, injected 2-3 config sweeps (monkeypatched _config_space) so the
suite stays fast; the full-space sweep is exercised by the llama block harness.
"""

import math

import pytest
import torch

import cute_gemm.gemm_runtime.autotune as epi_autotune
from cute_gemm.cute_dsl_utils import get_device_capacity
from cute_gemm.gemm_runtime.autotune import sink_arg_shapes, tuned_mod_gemm
from cute_gemm.epilogue.library import rms_fused, rstd_swiglu_epi
from cute_gemm.gemm_config import GemmConfig


def _cap():
    return get_device_capacity(torch.device("cuda"))[0]


def _cfg(tile_m, tile_n, cluster_m=1, pingpong=False):
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster_m,
        cluster_n=1,
        pingpong=pingpong,
        device_capacity=_cap(),
    )


@pytest.fixture()
def small_space(monkeypatch, tmp_path):
    import cute_gemm.cache

    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))  # hermetic winner cache
    # The artifact dir is captured from DLKERNEL_ARTIFACT_CACHE_DIR at package
    # import, so a setenv here is a no-op; patch the live attribute that
    # get_cache_path() reads on every call instead.
    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path / "artifacts"))
    cfgs = [_cfg(128, 128), _cfg(128, 256), _cfg(128, 192, pingpong=_cap() in (9, 12))]
    monkeypatch.setattr(epi_autotune, "_config_space", lambda mod, device: cfgs)
    monkeypatch.setattr(epi_autotune, "_MOD_TUNERS", {})
    return cfgs


def test_tuned_mod_gemm_matches_explicit(small_space):
    """Winner output is bitwise-equal to an explicit mod.gemm at the same
    config; sink buffers sized worst-case get sliced per config."""
    device = "cuda"
    torch.random.manual_seed(40)
    l, m, n, k = 2, 512, 1536, 736
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / math.sqrt(k) * 4
    weight = torch.randn((l, n), device=device, dtype=torch.float32)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    premult = torch.empty_like(D)
    shapes = sink_arg_shapes(rms_fused, m, n, l=l)
    assert shapes == {"sqsum": (l, m, (n + 127) // 128)}  # min tile_N in the space
    sqsum = torch.empty(shapes["sqsum"], device=device, dtype=torch.float32)

    res = tuned_mod_gemm(
        rms_fused, A, B, D, epi_args=dict(weight=weight, premult=premult, sqsum=sqsum)
    )
    n_tiles = (n + res.config.tile_n - 1) // res.config.tile_n
    assert res.sinks["sqsum"].shape == (l, m, n_tiles)

    D2 = torch.empty_like(D)
    premult2 = torch.empty_like(premult)
    sqsum2 = torch.empty((l, m, n_tiles), device=device, dtype=torch.float32)
    c = res.config
    rms_fused.gemm(
        A,
        B,
        D2,
        epi_args=dict(weight=weight, premult=premult2, sqsum=sqsum2),
        tile_M=c.tile_m,
        tile_N=c.tile_n,
        cluster_M=c.cluster_m,
        cluster_N=c.cluster_n,
        pingpong=c.pingpong,
    )
    assert torch.equal(D, D2)
    assert torch.equal(premult, premult2)
    assert torch.equal(res.sinks["sqsum"], sqsum2)
    # And the math is right regardless of which config won.
    x = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    sq_ref = x.pow(2).unflatten(-1, (n_tiles, c.tile_n)).sum(-1)
    assert (res.sinks["sqsum"] - sq_ref).abs().max().item() < 1e-3 * sq_ref.abs().max().item()


def test_tuned_mod_gemm_cache_hit(small_space):
    """Second call with identical metadata skips the sweep (in-memory cache)."""
    device = "cuda"
    torch.random.manual_seed(41)
    l, m, n, k = 2, 384, 1024, 512
    A = torch.randn((l, m, k), device=device, dtype=torch.bfloat16)
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16)
    D = torch.empty((l, m, n), device=device, dtype=torch.bfloat16)
    postact = torch.empty((l, m, n // 2), device=device, dtype=torch.bfloat16)
    rstd = torch.rand((l, m), device=device, dtype=torch.float32) + 0.5

    args = dict(rstd=rstd, postact=postact)
    tuned_mod_gemm(rstd_swiglu_epi, A, B, D, epi_args=args)
    tuner = next(iter(epi_autotune._MOD_TUNERS.values()))
    assert len(tuner.cache) == 1
    bench_time = tuner.bench_time
    tuned_mod_gemm(rstd_swiglu_epi, A, B, D, epi_args=args)
    assert len(tuner.cache) == 1 and tuner.bench_time == bench_time  # no re-bench


def test_prune_rules(small_space):
    """Host legality is independent of caller sink capacity.

    Sink buffers are validated once at the tuned-call boundary; capacity must
    not change the candidate set or therefore the shared cache winner.
    """
    if _cap() != 9:
        pytest.skip("prune-rule assertions are written for SM90")
    named = dict(
        A=torch.empty((384, 512), device="cuda", dtype=torch.bfloat16),
        B=torch.empty((1024, 512), device="cuda", dtype=torch.bfloat16),
        b_kn=False,
    )
    from cute_gemm.autotuner import AutotuneConfig

    confs = [AutotuneConfig(config=c) for c in small_space + [_cfg(128, 208)]]
    from cute_gemm.epilogue.legality import prune_epilogue_configs

    surv = prune_epilogue_configs(rstd_swiglu_epi, None, confs, named)
    assert all(c.kwargs["config"].tile_n % 32 == 0 for c in surv), [
        c.kwargs["config"].tile_n for c in surv
    ]
    # A partial buffer no longer prunes candidates. The tuned entry point
    # rejects it before the policy/cache path is reached.
    small_buf = torch.empty((384, 4), device="cuda", dtype=torch.float32)
    surv2 = prune_epilogue_configs(rms_fused, None, confs, named | dict(sqsum=small_buf))
    assert surv2 == prune_epilogue_configs(rms_fused, None, confs, named)


def test_full_sink_contract_rejects_partial_buffer(monkeypatch):
    """Undersized caller-owned sinks fail before tuning/cache lookup."""

    class Sink:
        dim = 0

        @staticmethod
        def sink_alloc_shape(lead, n, tile_m, tile_n, num_seqs=None):
            del tile_m, num_seqs
            return (lead[-1], (n + tile_n - 1) // tile_n)

    class Mod:
        sinks = {"stats": Sink()}

    config = GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=1,
        cluster_n=1,
        device_capacity=9,
        is_dynamic_persistent=False,
    )
    monkeypatch.setattr(epi_autotune, "_config_space", lambda mod, device: [config])
    A = torch.empty((256, 16))
    B = torch.empty((128, 16))
    with pytest.raises(ValueError, match="undersized.*shared-winner"):
        epi_autotune._validate_full_sink_buffers(
            Mod(),
            {"stats": torch.empty((128, 1))},
            A,
            B,
            b_kn=False,
            A_idx=None,
            cu_seqlens_m=None,
            n_gemm=128,
        )
    allocated = {}
    epi_autotune._validate_full_sink_buffers(
        Mod(),
        allocated,
        A,
        B,
        b_kn=False,
        A_idx=None,
        cu_seqlens_m=None,
        n_gemm=128,
    )
    assert tuple(allocated["stats"].shape) == (256, 1)


def test_tuned_wrapper_slices_sink_with_per_cta_tile_m():
    """SM100 2-CTA configs expose half of config.tile_m to each CTA sink."""

    class Sink:
        dim = 1

        @staticmethod
        def sink_alloc_shape(lead, n, tile_m, tile_n, num_seqs=None):
            return (tile_m,)

    class Mod:
        _ident = "sink_tile_probe"
        sinks = {"sink": Sink()}

        def __init__(self):
            self.seen_shape = None

        def gemm(self, A, B, D, C, *, epi_args, **kwargs):
            self.seen_shape = tuple(epi_args["sink"].shape)
            return "plan"

    mod = Mod()
    fn = epi_autotune._make_tuned_fn(mod, ("sink",))
    config = GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=2,
        cluster_n=1,
        device_capacity=10,
        is_dynamic_persistent=False,
    )

    result = fn(
        A=torch.empty((16, 4)),
        B=torch.empty((8, 4)),
        D=torch.empty((16, 8)),
        sink=torch.empty((128,)),
        config=config,
    )

    assert result == "plan"
    assert mod.seen_shape == (64,)


def test_epilogue_prune_sizes_varlen_m_sink_with_num_sequences(monkeypatch):
    """The pruning capacity check must use the same varlen allocation contract
    as frontend validation and the actual tuned wrapper."""

    class Sink:
        dim = 1
        check_oob = True

        @staticmethod
        def sink_alloc_shape(lead, n, tile_m, tile_n, num_seqs=None):
            assert num_seqs == 4
            return (lead[-1] // tile_m + num_seqs, n)

    class Mod:
        mode = "element"
        outputs = ()
        sinks = {"sink": Sink()}

    monkeypatch.setattr(epi_autotune, "get_device_capacity", lambda device: (10, 0))
    config = GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=2,
        cluster_n=1,
        device_capacity=10,
        is_dynamic_persistent=False,
    )
    conf = epi_autotune.AutotuneConfig(config=config)
    named = {
        "A": torch.empty((128, 16)),
        "B": torch.empty((32, 16)),
        "b_kn": False,
        "cu_seqlens_m": torch.zeros((5,), dtype=torch.int32),
        "sink": torch.empty((6, 32)),
    }

    from cute_gemm.epilogue.legality import prune_epilogue_configs

    assert prune_epilogue_configs(Mod(), None, [conf], named, device_capacity=(10, 0)) == [conf]


def test_mod_digest_in_disk_key(small_space):
    """The disk-cache directory hash includes the mod digest via key=; two
    mods with different fn bodies must not share tuning files."""
    from cute_gemm.epilogue.library import rms_fused as m1, rstd_swiglu_epi as m2

    t1 = epi_autotune._get_tuner(m1, ("weight",), False, torch.device("cuda"))
    t2 = epi_autotune._get_tuner(m2, ("rstd",), False, torch.device("cuda"))
    assert m1.semantic_digest != m2.semantic_digest
    assert t1.fn.__name__ != t2.fn.__name__
    assert "mod_digest" in t1.keys


# Module-level (importable digest anchors) for the transform-aware sweep.
from cute_gemm.epilogue.frontend import gemm_epilogue  # noqa: E402
from cute_gemm.operand_transform import a_transform  # noqa: E402


@gemm_epilogue()
def _tuner_ident_epi(acc):
    return {"D": acc}


@a_transform(vec_size=2)
def _tuner_halve_a(x):
    return x * 0.5


def test_tuned_value_transform(small_space):
    """A value transform tunes through the generic path: the transform digest
    keys the tuner and the winner computes (0.5 * A) @ B^T."""
    if _cap() != 9:
        pytest.skip("A transforms are SM90 RS-mainloop only")
    device = "cuda"
    torch.random.manual_seed(42)
    m, n, k = 256, 512, 256
    A = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    B = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    D = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    tuned_mod_gemm(_tuner_ident_epi, A, B, D, epi_args={}, transform_a=_tuner_halve_a)
    ref = (0.5 * A.float()) @ B.float().t()
    assert (D.float() - ref).abs().max().item() < 2e-2 * ref.abs().max().item()
    key = next(iter(epi_autotune._MOD_TUNERS))
    assert _tuner_halve_a.semantic_digest in key


def test_tuned_w4_transform(small_space):
    """A layout-owning transform tunes: config_ok prunes pingpong, the blob
    crosses as B, and the winner matches the dequant reference."""
    if _cap() != 9:
        pytest.skip("W4 transforms are SM90-only")
    from cute_gemm.operand_transform.formats import W4_FORMATS

    device = "cuda"
    torch.random.manual_seed(43)
    fmt = W4_FORMATS["int4"]
    n, k, m = 128, 512, 32
    w = torch.randn(n, k, device=device, dtype=torch.float32) * 0.05
    q, sf = fmt.quantize_reference(w)
    blob, sfb = fmt.prepare(q, sf)
    wd = fmt.dequant_reference(q, sf)
    act = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    D = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    res = tuned_mod_gemm(
        _tuner_ident_epi, act, blob, D, epi_args={}, transform_a="int4", transform_sf=sfb
    )
    assert not res.config.pingpong  # config_ok pruned the pingpong candidate
    ref = act.float() @ wd.float().t()
    assert (D.float() - ref).abs().max().item() < 2e-2 * ref.abs().max().item()


def test_varlen_epilogue_reuses_winner_within_bucket(small_space):
    """E1 end-to-end: two varlen_m calls in the same average-rows bucket share
    one tuned winner (exact shapes of D/sink buffers differ); a dense call on
    the same tuner stays on the default exact key."""
    device = "cuda"
    torch.random.manual_seed(43)
    l, n, k = 2, 512, 256
    B = torch.randn((l, n, k), device=device, dtype=torch.bfloat16) / 16
    weight = torch.randn((l, n), device=device, dtype=torch.float32)

    def run(total_m):
        # avg rows/expert = 1300 / 1350 -> both bucket 2048, same leaf.
        A = torch.randn((total_m, k), device=device, dtype=torch.bfloat16) / 16
        D = torch.empty((total_m, n), device=device, dtype=torch.bfloat16)
        premult = torch.empty_like(D)
        # Packed 2D varlen layout: no batch lead, sinks follow (m, n_tiles).
        shapes = sink_arg_shapes(rms_fused, total_m, n, num_seqs=l)
        sqsum = torch.empty(shapes["sqsum"], device=device, dtype=torch.float32)
        cu = torch.zeros(l + 1, device=device, dtype=torch.int32)
        for i in range(1, l + 1):
            cu[i] = total_m * i // l
        res = tuned_mod_gemm(
            rms_fused,
            A,
            B,
            D,
            epi_args=dict(weight=weight, premult=premult, sqsum=sqsum),
            cu_seqlens_m=cu,
        )
        return res, D, premult, sqsum

    res1, D1, premult1, sqsum1 = run(2 * 1300)
    tuner = next(iter(epi_autotune._MOD_TUNERS.values()))
    assert tuner._budget_override is True  # varlen epilogue opts into budget env
    assert len(tuner.cache) == 1
    bench_time = tuner.bench_time

    res2, _, _, _ = run(2 * 1350)
    assert len(tuner.cache) == 1, "same bucket must reuse the tuned winner"
    assert tuner.bench_time == bench_time  # no re-bench
    assert res2.config == res1.config

    # A dense call on the same tuner stays on the exact default key (m != n:
    # m == n would make the 'weight' operand's row/col inference ambiguous).
    m_dense = 384
    A = torch.randn((l, m_dense, k), device=device, dtype=torch.bfloat16) / 16
    D = torch.empty((l, m_dense, n), device=device, dtype=torch.bfloat16)
    premult = torch.empty_like(D)
    shapes = sink_arg_shapes(rms_fused, m_dense, n, l=l)
    sqsum = torch.empty(shapes["sqsum"], device=device, dtype=torch.float32)
    tuned_mod_gemm(rms_fused, A, B, D, epi_args=dict(weight=weight, premult=premult, sqsum=sqsum))
    assert len(tuner.cache) == 2  # dense keyed separately, budget not overridden
    assert tuner._budget_override is False
