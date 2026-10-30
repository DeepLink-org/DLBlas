# Copyright (c) 2025, DeepLink.
"""Large-shape SM90 GEMM + fused-epilogue tests for the vendored cute_gemm.

The CuTe-DSL library vendored under ``dlblas/kernels/cute_gemm`` is used
through the DLBlas namespace (``dlblas.cute_gemm``), so the tests exercise the
same entry point users get; the adapter falls back to the in-repo copy when
cute_gemm is not installed.

Every case runs the explicit ``GemmSm90`` class on large M/N/K and checks the
fused outputs against a float32 torch reference:

* ``gemm_with_activation``: elementwise (relu / gelu-tanh / silu) and gated
  (swiglu / reglu / geglu) activations, with bias and residual ``C`` fusion.
* epilogue-mod kernels bound to ``GemmSm90``: relu, scaled residual and the
  linear epilogue (alpha*acc + beta*C + rowvec + colvec).
"""

import math

import pytest
import torch
import torch.nn.functional as F

try:
    import dlblas.cute_gemm as cute
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
    pytest.skip(f"dlblas.cute_gemm is unavailable: {exc}", allow_module_level=True)

TuneMode = cute.TuneMode
GemmBase = cute.GemmBase
GemmSm90 = cute.GemmSm90
linear_epi = cute.linear_epi
relu_mod = cute.relu_mod
scaled_residual = cute.scaled_residual

sm90_only = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires an SM90 (H100/H200) GPU",
)

_TOL = {torch.bfloat16: 3e-2, torch.float16: 2e-2}
_ACT = {
    "relu": torch.relu,
    "silu": F.silu,
    "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
}
_GATED = {
    "swiglu": lambda gate, up: F.silu(gate) * up,
    "reglu": lambda gate, up: torch.relu(gate) * up,
    "geglu": lambda gate, up: F.gelu(gate, approximate="tanh") * up,
}


def _rel_check(out, ref, name, tol):
    err = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert (
        err <= tol * scale + 1e-2
    ), f"{name}: max abs err {err:.3e}, ref scale {scale:.3e}"


def _assert_sm90():
    # The fused kernels must resolve to the SM90 implementation, not a fallback.
    assert GemmBase.select_gemm_class(9) is GemmSm90


@sm90_only
@pytest.mark.parametrize("m,k,n", [(8192, 4096, 8192), (2048, 16384, 2048)])
@pytest.mark.parametrize(
    "activation,dtype",
    [
        ("relu", torch.bfloat16),
        ("relu", torch.float16),
        ("gelu_tanh_approx", torch.bfloat16),
        ("silu", torch.bfloat16),
    ],
)
def test_gemm_sm90_activation_large(activation, dtype, m, k, n, tmp_path, monkeypatch):
    """alpha*acc + C + bias then relu / gelu-tanh / silu, on large shapes."""
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    _assert_sm90()
    torch.manual_seed(0)
    alpha = 0.5
    A = torch.randn(m, k, device="cuda", dtype=dtype) / math.sqrt(k)
    B = torch.randn(k, n, device="cuda", dtype=dtype) / math.sqrt(k)
    C = torch.randn(m, n, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)

    preact, postact = GemmSm90.gemm_with_activation(
        A,
        B,
        C=C,
        bias=bias,
        activation=activation,
        alpha=alpha,
        tune_mode=TuneMode.STATIC,
    )

    ref_pre = alpha * (A.float() @ B.float()) + C.float() + bias.float()
    ref_post = _ACT[activation](ref_pre)
    assert preact.shape == (m, n) and preact.dtype == dtype
    assert postact.shape == (m, n) and postact.dtype == dtype
    _rel_check(preact, ref_pre, "preact", _TOL[dtype])
    _rel_check(postact, ref_post, "postact", _TOL[dtype])


@sm90_only
@pytest.mark.parametrize(
    "activation,dtype",
    [
        ("swiglu", torch.bfloat16),
        ("swiglu", torch.float16),
        ("reglu", torch.bfloat16),
        ("geglu", torch.bfloat16),
    ],
)
def test_gemm_sm90_gated_large(activation, dtype, tmp_path, monkeypatch):
    """Gated activations: interleaved gate/up columns, N = 2 * up width."""
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    _assert_sm90()
    torch.manual_seed(1)
    m, k, n_up = 8192, 4096, 4096
    n = 2 * n_up
    A = torch.randn(m, k, device="cuda", dtype=dtype) / math.sqrt(k)
    B = torch.randn(k, n, device="cuda", dtype=dtype) / math.sqrt(k)

    preact, postact = GemmSm90.gemm_with_activation(
        A, B, activation=activation, tune_mode=TuneMode.STATIC
    )

    ref_pre = A.float() @ B.float()
    gate, up = ref_pre[..., 0::2], ref_pre[..., 1::2]
    ref_post = _GATED[activation](gate, up)
    assert preact.shape == (m, n) and postact.shape == (m, n_up)
    _rel_check(preact, ref_pre, "preact", _TOL[dtype])
    _rel_check(postact, ref_post, "postact", _TOL[dtype])


@sm90_only
def test_gemm_sm90_batched_residual_bias_large(tmp_path, monkeypatch):
    """Batched (L=2) fused GEMM with residual C and per-batch bias."""
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    _assert_sm90()
    torch.manual_seed(2)
    l, m, k, n = 2, 4096, 4096, 8192
    alpha = 0.5
    A = torch.randn(l, m, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn(l, k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    C = torch.randn(l, m, n, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(l, n, device="cuda", dtype=torch.bfloat16)

    preact, postact = GemmSm90.gemm_with_activation(
        A,
        B,
        C=C,
        bias=bias,
        activation="relu",
        alpha=alpha,
        tune_mode=TuneMode.STATIC,
    )
    ref_pre = (
        alpha * torch.bmm(A.float(), B.float()) + C.float() + bias.float().unsqueeze(-2)
    )
    tol = _TOL[torch.bfloat16]
    assert preact.shape == (l, m, n) and postact.shape == (l, m, n)
    _rel_check(preact, ref_pre, "preact", tol)
    _rel_check(postact, torch.relu(ref_pre), "postact", tol)

    # store_preact=False: the fused pre-activation stays off the output list.
    preact2, postact2 = GemmSm90.gemm_with_activation(
        A,
        B,
        C=C,
        bias=bias,
        activation="relu",
        alpha=alpha,
        store_preact=False,
        tune_mode=TuneMode.STATIC,
    )
    assert preact2 is None
    _rel_check(postact2, torch.relu(ref_pre), "postact (no preact)", tol)


@sm90_only
@pytest.mark.parametrize("mod_name", ["relu", "scaled_residual", "linear_epi"])
def test_gemm_sm90_epi_mod_large(mod_name, tmp_path, monkeypatch):
    """Epilogue-mod kernels bound to GemmSm90 on large batched shapes."""
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    _assert_sm90()
    torch.manual_seed(3)
    l, m, k, n = 2, 4096, 4096, 8192
    dtype = torch.bfloat16
    tol = _TOL[dtype]
    A = torch.randn(l, m, k, device="cuda", dtype=dtype) / math.sqrt(k)
    # Epilogue mods take the kernel-native (n, k) B layout.
    B = torch.randn(l, n, k, device="cuda", dtype=dtype) / math.sqrt(k)
    acc = torch.einsum("lmk,lnk->lmn", A.float(), B.float())
    D = torch.empty(l, m, n, device="cuda", dtype=dtype)
    kw = dict(tile_M=128, tile_N=256, cluster_M=2, cluster_N=1)

    if mod_name == "relu":
        postact = torch.empty_like(D)
        GemmSm90.bind_epilogue(relu_mod).gemm(
            A, B, D, epi_args=dict(postact=postact), **kw
        )
        _rel_check(D, acc, "D", tol)
        _rel_check(postact, torch.relu(acc), "postact", tol)
    elif mod_name == "scaled_residual":
        C = torch.randn(l, m, n, device="cuda", dtype=dtype)
        alpha = 0.5
        GemmSm90.bind_epilogue(scaled_residual).gemm(
            A, B, D, C, epi_args=dict(alpha=alpha), **kw
        )
        _rel_check(D, alpha * acc + C.float(), "D", tol)
    else:
        # bias_n / bias_m row-vs-col inference needs distinct m and n extents.
        assert m != n
        C = torch.randn(l, m, n, device="cuda", dtype=dtype)
        alpha, beta = 1.5, 0.5
        bias_n = torch.randn(l, n, device="cuda", dtype=torch.float32)
        bias_m = torch.randn(l, m, device="cuda", dtype=torch.float32)
        GemmSm90.bind_epilogue(linear_epi).gemm(
            A,
            B,
            D,
            C,
            epi_args=dict(alpha=alpha, beta=beta, bias_n=bias_n, bias_m=bias_m),
            **kw,
        )
        ref = (
            alpha * acc + beta * C.float() + bias_n.unsqueeze(-2) + bias_m.unsqueeze(-1)
        )
        _rel_check(D, ref, "D", tol)
