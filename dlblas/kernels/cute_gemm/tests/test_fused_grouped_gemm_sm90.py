"""SM90 test for the two MoE fused grouped-GEMM primitives.

Granularity follows the FlashInfer house style (tests/gemm, tests/grouped_mm):
one reference sweep per axis with representative combos, one named test per
behavioral edge, and CPU-side contract tests for validation and identity.

Up-projection (preact + postact) feeds down-projection (dH, A-prime, dS)
against an independent fp32 reference, stopping at the primitive boundary.
Named edge tests pin routing shapes, bias/preact modes, tile crossings, the
tuned (bucketed) path, adversarial switch combinations and balanced shapes at
production dimensions, and interleaved-shape reuse with no retained tensors.
Contract tests pin the MoE-seam K=0 boundaries, CPU expert_offsets
validation, the down-projection unpack, class selection, the kernel_variant
registry, and the role identity wiring (compiled token, minted MRO, cold
subprocess, async-compile worker).
"""

import pytest
import torch
import torch.nn.functional as F

from cute_gemm import TuneMode, moe_down_proj_dact, moe_up_proj_act
from cute_gemm.fused_grouped_gemm.gemm_sm90 import GemmMoeSm90
from cute_gemm.gemm_base import GemmBase

_ATOL = 0.20
_RTOL = 0.08

# The role's own class constant: which activations take interleaved [g, u] W13.
_GATED_ACTIVATIONS = GemmMoeSm90._GATED_ACTIVATIONS


def _assert_close(actual, expected, name, dtype=None):
    # dtype is accepted and ignored: signature parity with _assert_close_scaled.
    torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL, rtol=_RTOL, msg=name)


def _assert_close_scaled(actual, expected, name, dtype):
    """Compare against the fp32 oracle with a per-dtype, magnitude-scaled budget.

    Residual sources: fp32 accumulation order over K, activation nonlinearity,
    and the 16-bit output store. atol scales with the reference's same-dtype
    magnitude with NO constant floor — the retired max(1.0, ..) floor pinned
    atol above the whole 0.02-scaled production references, letting an
    all-zero output compare equal. fp16 stores get the tighter budget
    (mantissa 2^-11 vs bf16 2^-8; the MMA is fp32-accumulating in both), and
    so does dtype=torch.float32, the dS-only case: the oracle re-reads its
    fp32 preact where the kernel consumed the 16-bit store.
    """
    scale = float(expected.to(dtype).float().abs().max())
    if dtype == torch.bfloat16:
        atol, rtol = 0.05 * scale, 0.05
    else:  # fp16 stores; fp32 is dS, residual set by the 16-bit preact it read
        atol, rtol = 0.02 * scale, 0.03
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol, msg=name)


def _assert_outputs(
    outputs, expected, tag, dtype, *, d_score_dtype=None, close=_assert_close_scaled
):
    """Compare one up->down round trip against the oracle, surface by surface.

    ``outputs``/``expected`` are prefixes of (preact, postact, dH, A-prime,
    dS) in that order; a None oracle entry (no stored preact, no router-grad
    reduce) asserts the kernel surface is None. dS compares under
    ``d_score_dtype`` when given, else the activation dtype.
    """
    for name, actual, exp in zip(
        ("preactivation", "postactivation", "dH", "A-prime", "dS"), outputs, expected
    ):
        if exp is None:
            assert actual is None, f"{tag} {name}"
        else:
            budget = d_score_dtype if name == "dS" and d_score_dtype is not None else dtype
            close(actual, exp, f"{tag} {name}", budget)


def _balanced_counts(routes, experts):
    """Near-uniform per-expert route counts: base everywhere, +1 for the first rem."""
    base, rem = divmod(routes, experts)
    counts = torch.full((experts,), base, dtype=torch.int64)
    counts[:rem] += 1
    return counts


def _offsets_from_counts(counts, device=None):
    """Exclusive int32 prefix offsets from per-expert counts."""
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).to(torch.int32).to(offsets.device)
    return offsets


def _interleave_rows(t, experts, intermediate):
    """Concat [all gate; all up] rows reordered to [g0, u0, g1, ...]; any trailing dims."""
    trailing = t.shape[2:]
    return (
        t.view(experts, 2, intermediate, *trailing)
        .transpose(1, 2)
        .reshape(experts, 2 * intermediate, *trailing)
    )


def _interleaved_to_concat(t, intermediate):
    """Feature-dim inverse of the interleaved storage: [g0, u0, g1, u1, ...] -> [gate; up]."""
    return torch.cat((t[..., 0::2], t[..., 1::2]), dim=-1)


def _gated_forward(activation, gate, up):
    return {
        "swiglu": lambda g, u: F.silu(g) * u,
        "geglu": lambda g, u: F.gelu(g, approximate="tanh") * u,
        "reglu": lambda g, u: F.relu(g) * u,
    }[activation](gate, up)


def _ungated_forward(activation, values):
    return {
        "silu": F.silu,
        "relu": F.relu,
        "relu_sq": lambda x: F.relu(x).square(),
        "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
    }[activation](values)


def _activation_forward(activation, preact_f32):
    """fp32 postactivation of an interleaved (gated) or natural (ungated) preact."""
    if activation in _GATED_ACTIVATIONS:
        return _gated_forward(activation, preact_f32[..., ::2], preact_f32[..., 1::2])
    return _ungated_forward(activation, preact_f32)


def _postact_and_grad(activation, preact):
    """Postactivation and its elementwise derivative via autograd (float32).

    For gated activations the preact columns are interleaved [g0, u0, g1, ...];
    differentiating through the strided views yields dH in the same layout.
    """
    z = preact.float().clone().requires_grad_(True)
    out = _activation_forward(activation, z)
    grad = torch.autograd.grad(out, z, torch.ones_like(out))[0]
    return out.detach(), grad


def _up_reference(x, weight, bias, offsets, token_indices, activation):
    """Gather -> per-expert linear(+bias) -> activation, in float32.

    ``weight`` is the concat [all gate; all up] storage for gated activations;
    the returned preact is interleaved [g0, u0, g1, ...] for those, natural
    order otherwise.
    """
    preacts, postacts = [], []
    for expert in range(weight.shape[0]):
        begin, end = int(offsets[expert]), int(offsets[expert + 1])
        values = F.linear(x[token_indices[begin:end].long()].float(), weight[expert].float())
        if bias is not None:
            values = values + bias[expert].float()
        if activation in _GATED_ACTIVATIONS:
            intermediate = weight.shape[1] // 2
            gate, up = values[:, :intermediate], values[:, intermediate:]
            # Explicit width: reshape(0, -1) is ambiguous for the empty segment.
            preacts.append(
                torch.stack((gate, up), dim=-1).reshape(values.shape[0], 2 * intermediate)
            )
        else:
            preacts.append(values)
        postacts.append(_activation_forward(activation, preacts[-1]))
    return torch.cat(preacts, dim=0), torch.cat(postacts, dim=0)


def _down_reference(grad_output, down_weight, preact, offsets, token_indices, scores, activation):
    d_h = torch.empty_like(preact)
    weighted_postact = torch.empty(
        preact.shape[0], down_weight.shape[-1], device=preact.device, dtype=preact.dtype
    )
    d_score = (
        torch.empty(preact.shape[0], device=preact.device, dtype=torch.float32)
        if scores is not None
        else None
    )
    for expert in range(down_weight.shape[0]):
        begin, end = int(offsets[expert]), int(offsets[expert + 1])
        rows = token_indices[begin:end].long()
        accumulator = grad_output[rows].float() @ down_weight[expert].float()
        postact, jac = _postact_and_grad(activation, preact[begin:end])
        score = (
            scores[begin:end].float()
            if scores is not None
            else torch.ones(end - begin, device=preact.device)
        )
        dout = score[:, None] * accumulator
        if activation in _GATED_ACTIVATIONS:
            # dH lives in preact space; each gate/up column pair shares one
            # postact-column gradient (jac's even/odd columns are d_gate/d_up).
            dout = dout.repeat_interleave(2, dim=1)
        d_h[begin:end] = (dout * jac).to(preact.dtype)
        weighted_postact[begin:end] = (score[:, None] * postact).to(preact.dtype)
        # The score derivative intentionally excludes score itself.
        if d_score is not None:
            d_score[begin:end] = (postact * accumulator).sum(dim=-1)
    return d_h, weighted_postact, d_score


def test_assert_close_scaled_rejects_zero_and_misrouting():
    """Comparator self-check: zero output and neighbor-expert rows must FAIL.

    Discrimination proof on the production fixture regime (inputs scaled by
    0.02, references far below 1.0) where the retired max(1.0, ..) floor held
    atol above the whole reference and an all-zero kernel output compared
    equal. Inputs mirror the engine defaults on qwen35_35b at tokens=123;
    reference-only, so the check needs no GPU.
    """
    model, tokens = "qwen35_35b", 123
    hidden, intermediate, experts, topk = _PRODUCTION_SHAPES[model]
    torch.manual_seed(sum(map(ord, model)) + tokens)
    routes = tokens * topk
    offsets = _offsets_from_counts(_balanced_counts(routes, experts))
    gen = torch.Generator().manual_seed(31 + tokens)
    token_indices = torch.randint(tokens, (routes,), generator=gen, dtype=torch.int32)
    scores = torch.linspace(0.125, 1.0, routes, dtype=torch.float32)
    dtype = torch.bfloat16
    x = 0.02 * torch.randn(tokens, hidden, dtype=dtype)
    gate_up_weight = 0.02 * torch.randn(experts, 2 * intermediate, hidden, dtype=dtype)
    gate_up_bias = 0.02 * torch.randn(experts, 2 * intermediate, dtype=dtype)
    down_weight = 0.02 * torch.randn(experts, hidden, intermediate, dtype=dtype)
    grad_output = 0.02 * torch.randn(tokens, hidden, dtype=dtype)

    expected_preact, expected_postact = _up_reference(
        x, gate_up_weight, gate_up_bias, offsets, token_indices, "swiglu"
    )
    _, _, expected_ds = _down_reference(
        grad_output, down_weight, expected_preact, offsets, token_indices, scores, "swiglu"
    )
    # Regime pin: the flagged references sit below the retired absolute floor
    # (0.05 bf16 / 0.02 fp16) — and the retired budget demonstrably admitted
    # an all-zero postact on this exact fixture, which is the blindness the
    # scale-anchored budget removes.
    assert float(expected_postact.abs().max()) < 0.02
    assert float(expected_ds.abs().max()) < 0.02
    torch.testing.assert_close(
        torch.zeros_like(expected_postact), expected_postact, atol=0.05, rtol=0.05
    )
    # (a) An all-zero output — the kernel-computed-nothing failure mode — must
    # fail on a bf16 surface and under the dedicated fp32 dS budget alike.
    with pytest.raises(AssertionError):
        _assert_close_scaled(
            torch.zeros_like(expected_postact), expected_postact, "zero postact", dtype
        )
    with pytest.raises(AssertionError):
        _assert_close_scaled(torch.zeros_like(expected_ds), expected_ds, "zero dS", torch.float32)
    # (b) Mis-routing — one expert's rows computed with the neighbor's weights.
    # Rows outside the swapped pair stay exact, so the comparator must catch
    # the mismatched segment on its own.
    swapped_weight = gate_up_weight.clone()
    swapped_weight[[0, 1]] = swapped_weight[[1, 0]]
    swapped_bias = gate_up_bias.clone()
    swapped_bias[[0, 1]] = swapped_bias[[1, 0]]
    misrouted_preact, misrouted_postact = _up_reference(
        x, swapped_weight, swapped_bias, offsets, token_indices, "swiglu"
    )
    with pytest.raises(AssertionError):
        _assert_close_scaled(misrouted_preact, expected_preact, "misrouted preact", dtype)
    with pytest.raises(AssertionError):
        _assert_close_scaled(misrouted_postact, expected_postact, "misrouted postact", dtype)


def _require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability(0)[0] != 9:
        pytest.skip("SM90 only")
    return torch.device("cuda")


def _purge_role_minted_classes():
    """Drop role-minted kernel classes so MRO assertions prove a FRESH mint.

    Only safe with a jit-cache miss guaranteed afterwards; warm plan caches
    short-circuit minting entirely, so the compiled-path tests spy on the
    token seam instead (see _record_kernel_variants).
    """
    import cute_gemm.epilogue.frontend as frontend_module
    from cute_gemm.gemm_runtime.identity import TORCH_OP_EPI_MODS

    for name, cls in list(vars(frontend_module).items()):
        if name.startswith("GemmMod") and isinstance(cls, type) and GemmMoeSm90 in cls.__mro__:
            delattr(frontend_module, name)
    for mod in TORCH_OP_EPI_MODS.values():
        mod._minted.clear()


def _record_kernel_variants(monkeypatch):
    """Spy on the class/token seam: the wire value observed per execution.

    Both the classmethod and the neutral registry's resolver seam are
    recorded on every call — warm or cold, eager or inside a custom-op body —
    proving the compiled call carried the role's variant token, immune to
    minted-class staleness or plan-cache reuse.
    """
    import cute_gemm.gemm_runtime.identity as identity_module

    recorded = []
    real_classmethod = GemmBase.__dict__["resolve_kernel_variant"].__func__

    def class_spy(cls, kernel_variant):
        recorded.append(kernel_variant)
        return real_classmethod(cls, kernel_variant)

    real_resolver = identity_module.resolve_kernel_variant_class

    def token_spy(token):
        recorded.append(token)
        return real_resolver(token)

    monkeypatch.setattr(GemmBase, "resolve_kernel_variant", classmethod(class_spy))
    monkeypatch.setattr(identity_module, "resolve_kernel_variant_class", token_spy)
    return recorded


def _run_moe_up_down(
    activation,
    layout,
    dtype,
    *,
    tokens,
    hidden,
    intermediate,
    experts,
    offsets,
    token_indices,
    seed=7,
    bias_dtype="activation",
    with_scores=True,
    store_preact=True,
):
    """One gather-mode up->down MoE round trip against the fp32 oracle.

    Runs both primitives at ``tune_mode=TuneMode.STATIC`` with scaled per-dtype tolerances
    and returns the kernel outputs for case-specific follow-up assertions.
    ``bias_dtype``: "activation" (default), a torch dtype (fp32 exercises the
    wide-bias epilogue port), or None (bias-free). ``token_indices=None``
    packs the activation rows expert-major instead of gathering.
    """
    device = torch.device("cuda")
    torch.manual_seed(seed)
    gated = activation in _GATED_ACTIVATIONS
    width = 2 * intermediate if gated else intermediate
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    gate_up_weight = torch.randn(experts, width, hidden, device=device, dtype=dtype)
    if bias_dtype is None:
        gate_up_bias = None
    else:
        bias = dtype if bias_dtype == "activation" else bias_dtype
        gate_up_bias = torch.randn(experts, width, device=device, dtype=bias)
    down_weight = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype)
    grad_output = torch.randn(tokens, hidden, device=device, dtype=dtype)
    offsets = torch.as_tensor(offsets, device=device, dtype=torch.int32)
    if token_indices is not None:
        token_indices = torch.as_tensor(token_indices, device=device, dtype=torch.int32)
    route_count = token_indices.shape[0] if token_indices is not None else x.shape[0]
    scores = torch.randn(route_count, device=device, dtype=torch.float32) if with_scores else None

    if layout == "interleaved":
        # Same values as concat storage, rows reordered to [g0, u0, g1, u1, ...].
        kernel_weight = _interleave_rows(gate_up_weight, experts, intermediate)
        kernel_bias = (
            None if gate_up_bias is None else _interleave_rows(gate_up_bias, experts, intermediate)
        )
    else:
        kernel_weight, kernel_bias = gate_up_weight, gate_up_bias

    postact, preact = moe_up_proj_act(
        x,
        kernel_weight,
        offsets,
        activation=activation,
        gate_up_layout=layout,
        token_indices=token_indices,
        gate_up_bias=kernel_bias,
        store_preact=store_preact,
        tune_mode=TuneMode.STATIC,
    )
    gather_indices = (
        token_indices
        if token_indices is not None
        else torch.arange(x.shape[0], device=device, dtype=torch.int64)
    )
    expected_preact, expected_postact = _up_reference(
        x, gate_up_weight, gate_up_bias, offsets, gather_indices, activation
    )
    # The down projection consumes the interleaved preact: the kernel's own
    # store-rounded output when available, else the oracle cast to the dtype.
    down_preact = preact if store_preact else expected_preact.to(dtype)
    d_h, weighted_postact, d_score = moe_down_proj_dact(
        grad_output,
        down_weight,
        down_preact,
        offsets,
        activation=activation,
        token_indices=token_indices,
        router_weights=scores,
        compute_router_weight_grad=with_scores,
        tune_mode=TuneMode.STATIC,
    )
    expected_dh, expected_a, expected_ds = _down_reference(
        grad_output, down_weight, down_preact, offsets, gather_indices, scores, activation
    )
    _assert_outputs(
        (preact, postact, d_h, weighted_postact, d_score),
        (
            expected_preact if store_preact else None,
            expected_postact,
            expected_dh,
            expected_a,
            expected_ds,
        ),
        f"{activation}/{layout}/{str(dtype).removeprefix('torch.')}",
        dtype,
    )
    return preact, postact, d_h, weighted_postact, d_score


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("activation", "gather", "reduce"),
    [
        ("swiglu", True, True),
        ("swiglu", False, False),
        ("relu_sq", True, False),
        ("relu_sq", False, True),
    ],
)
def test_moe_role_fullgraph_reference(activation, gather, reduce):
    """torch.compile(fullgraph=True) MoE up->down vs an independent reference.

    The role crosses the compiled boundary as the kernel_variant string and
    must keep numerics across all four gather x reduce corners (one corner
    per activation here; gated/ungated covered by the activation axis).
    fullgraph=True errors on any graph break, so passing is also the
    no-break proof for the codec path.
    """
    device = _require_sm90()
    torch.manual_seed(21)
    tokens, hidden, intermediate, experts = 4, 16, 8, 3
    width = 2 * intermediate if activation == "swiglu" else intermediate
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(experts, width, hidden, device=device, dtype=torch.bfloat16)
    gate_up_bias = torch.randn(experts, width, device=device, dtype=torch.bfloat16)
    down_weight = torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16)
    grad_output = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    scores = torch.tensor([0.0, 0.25, -0.5, 1.25], device=device, dtype=torch.float32)
    offsets = torch.tensor([0, 2, 2, 4], device=device, dtype=torch.int32)
    token_indices = torch.tensor([3, 0, 2, 3], device=device, dtype=torch.int32) if gather else None
    # The gather-free reference still indexes rows segment-by-segment.
    gather_indices = token_indices if gather else torch.arange(tokens, device=device)

    def block(x, w13, b13, w3, go, sc, offs, idx):
        postact, preact = moe_up_proj_act(
            x,
            w13,
            offs,
            activation=activation,
            gate_up_layout="concat",
            token_indices=idx,
            gate_up_bias=b13,
            store_preact=True,
            tune_mode=TuneMode.STATIC,
        )
        d_h, weighted_postact, d_score = moe_down_proj_dact(
            go,
            w3,
            preact,
            offs,
            activation=activation,
            token_indices=idx,
            router_weights=sc,
            compute_router_weight_grad=reduce,
            tune_mode=TuneMode.STATIC,
        )
        return preact, postact, d_h, weighted_postact, d_score

    fn = torch.compile(block, fullgraph=True)
    preact, postact, d_h, weighted_postact, d_score = fn(
        x,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        grad_output,
        scores if reduce else None,
        offsets,
        token_indices,
    )

    expected_preact, expected_postact = _up_reference(
        x, gate_up_weight, gate_up_bias, offsets, gather_indices, activation
    )
    expected_dh, expected_a, expected_ds = _down_reference(
        grad_output,
        down_weight,
        expected_preact,
        offsets,
        gather_indices,
        scores if reduce else None,
        activation,
    )
    _assert_outputs(
        (preact, postact, d_h, weighted_postact, d_score),
        (expected_preact, expected_postact, expected_dh, expected_a, expected_ds),
        f"{activation}/gather={gather}",
        torch.bfloat16,
        close=_assert_close,
    )
    if reduce:
        assert d_score.dtype == torch.float32


@pytest.mark.parametrize(
    ("score_dtype", "gather", "reduce"),
    [
        (torch.bfloat16, False, False),
        (torch.bfloat16, True, True),
        (torch.float16, True, True),
        (torch.float8_e4m3fn, False, True),
        (torch.float8_e4m3fn, True, False),
        (torch.float8_e5m2, True, True),
    ],
)
def test_moe_down_lowbit_router_weights(score_dtype, gather, reduce):
    device = _require_sm90()
    torch.manual_seed(91)
    r, h, i = 259, 256, 128
    dtype = torch.float16 if score_dtype == torch.float16 else torch.bfloat16
    grad_output = torch.randn(113 if gather else r, h, device=device, dtype=dtype)
    down_weight = torch.randn(3, h, i, device=device, dtype=dtype) / 16
    preact = torch.randn(r, 2 * i, device=device, dtype=dtype)
    offsets = torch.tensor([0, 65, 65, r], device=device, dtype=torch.int32)
    scores = torch.randn(r, device=device).to(score_dtype)
    indices = torch.randint(113, (r,), device=device, dtype=torch.int32) if gather else None
    got = moe_down_proj_dact(
        grad_output,
        down_weight,
        preact,
        offsets,
        router_weights=scores,
        token_indices=indices,
        compute_router_weight_grad=reduce,
        tune_mode=TuneMode.STATIC,
    )
    rows = indices if gather else torch.arange(r, device=device)
    ref = _down_reference(grad_output, down_weight, preact, offsets, rows, scores, "swiglu")
    for name, actual, expected in zip(("dH", "postact", "dS"), got, ref):
        if name == "dS" and not reduce:
            assert actual is None
        else:
            _assert_close_scaled(actual, expected, name, torch.float32 if name == "dS" else dtype)


@pytest.mark.parametrize("score_dtype", [torch.float64, torch.int32])
def test_moe_down_rejects_unsupported_router_dtype(score_dtype):
    """Router dtypes outside _ROUTER_DTYPES fail validation before any GPU work."""
    device = _require_sm90()
    r, h, i = 16, 32, 16
    grad_output = torch.randn(r, h, device=device, dtype=torch.bfloat16)
    down_weight = torch.randn(3, h, i, device=device, dtype=torch.bfloat16)
    preact = torch.randn(r, 2 * i, device=device, dtype=torch.bfloat16)
    offsets = torch.tensor([0, r, r, r], device=device, dtype=torch.int32)
    scores = torch.ones(r, device=device, dtype=score_dtype)
    with pytest.raises(TypeError, match="router_weights must use one of"):
        moe_down_proj_dact(grad_output, down_weight, preact, offsets, router_weights=scores)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_fp32_router_weights_with_odd_offsets(dtype):
    """The supported FP32 scale path keeps numerical parity on ragged segments."""
    _require_sm90()
    _run_moe_up_down(
        "swiglu",
        "concat",
        dtype,
        tokens=129,
        hidden=256,
        intermediate=128,
        experts=3,
        offsets=[0, 65, 65, 129],
        token_indices=None,
    )


@pytest.mark.parametrize("score_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_moe_down_lowbit_fullgraph(score_dtype):
    device = _require_sm90()
    torch.manual_seed(92)
    r, h, i = 129, 256, 128
    go = torch.randn(r, h, device=device, dtype=torch.bfloat16)
    weight = torch.randn(3, h, i, device=device, dtype=go.dtype) / 16
    preact = torch.randn(r, 2 * i, device=device, dtype=go.dtype)
    offsets = torch.tensor([0, 65, 65, r], device=device, dtype=torch.int32)
    scores = torch.randn(r, device=device).to(score_dtype)

    def call(scores):
        return moe_down_proj_dact(
            go,
            weight,
            preact,
            offsets,
            router_weights=scores,
            compute_router_weight_grad=True,
            tune_mode=TuneMode.STATIC,
        )

    actual = torch.compile(call, fullgraph=True)(scores)
    reference = _down_reference(
        go, weight, preact, offsets, torch.arange(r, device=device), scores, "swiglu"
    )
    for name, got, want in zip(("dH", "postact", "dS"), actual, reference):
        _assert_close_scaled(got, want, name, torch.float32 if name == "dS" else go.dtype)


@pytest.mark.parametrize(
    "spec",
    [
        (256, 128, 1, 2, False),
        (128, 224, 1, 2, False),
        (128, 160, 2, 1, True),
        (192, 128, 1, 2, True),
    ],
)
def test_moe_down_lowbit_reported_configs(spec):
    """Pin every reported failing tile so tuning cannot route around the bug."""
    from cute_gemm.gemm_config import GemmConfig

    device = _require_sm90()
    torch.manual_seed(93)
    r, h, i = 129, 256, 256
    go = torch.randn(r, h, device=device, dtype=torch.bfloat16)
    weight = torch.randn(3, h, i, device=device, dtype=go.dtype) / 16
    preact = torch.randn(r, 2 * i, device=device, dtype=go.dtype)
    offsets = torch.tensor([0, 65, 65, r], device=device, dtype=torch.int32)
    scores = torch.randn(r, device=device, dtype=go.dtype)
    m, n, cm, cn, pp = spec
    actual = GemmMoeSm90.gemm_with_activation_grad(
        go,
        weight,
        preact,
        activation="swiglu",
        colvec_scale=scores,
        colvec_reduce=True,
        cu_seqlens_m=offsets,
        dynamic_scheduler=True,
        config=GemmConfig(tile_m=m, tile_n=n, cluster_m=cm, cluster_n=cn, pingpong=pp),
    )
    reference = _down_reference(
        go, weight, preact, offsets, torch.arange(r, device=device), scores, "swiglu"
    )
    for name, got, want in zip(("dH", "postact", "dS"), actual, reference):
        _assert_close_scaled(got, want, name, torch.float32 if name == "dS" else go.dtype)


def test_moe_down_unpack_contract(monkeypatch):
    """moe_down_proj_dact unpacks gemm_with_activation_grad strictly per the flag.

    A wrong-arity return is a contract violation that must surface as a
    RuntimeError at the boundary, never be adapted away by None-filling. CPU
    tensors suffice: validation passes and the patched class entry never
    launches anything. The spy patches GemmBase.gemm_with_activation_grad —
    the class entry the role method calls — so every call is intercepted, and
    the recorded receiver class proves the entry was invoked as the role's
    own.
    """
    grad_output = torch.randn(4, 16, dtype=torch.bfloat16)
    down_weight = torch.randn(3, 16, 8, dtype=torch.bfloat16)
    preact = torch.randn(4, 16, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
    router_weights = torch.ones(4, dtype=torch.float32)
    # zeros, not empty: torch.equal is NaN-unequal, and empty's garbage memory
    # can contain NaN bit patterns — that made this control flaky.
    dx = torch.zeros(4, 16, dtype=torch.bfloat16)
    postact = torch.zeros(4, 8, dtype=torch.bfloat16)
    d_score = torch.zeros(4, dtype=torch.float32)
    seen_classes = []

    def fake_2(cls, grad_output, down_weight, preact, **kwargs):
        seen_classes.append(cls)
        return dx.clone(), postact.clone()

    def fake_3_none(cls, grad_output, down_weight, preact, **kwargs):
        seen_classes.append(cls)
        return dx.clone(), postact.clone(), None

    def fake_3(cls, grad_output, down_weight, preact, **kwargs):
        seen_classes.append(cls)
        return dx.clone(), postact.clone(), d_score.clone()

    def call(flag):
        return moe_down_proj_dact(
            grad_output,
            down_weight,
            preact,
            offsets,
            activation="swiglu",
            router_weights=router_weights if flag else None,
            compute_router_weight_grad=flag,
            tune_mode=TuneMode.STATIC,
        )

    def install(fake):
        monkeypatch.setattr(GemmBase, "gemm_with_activation_grad", classmethod(fake), raising=False)

    for fake, flag, error in (
        (fake_2, True, "compute_router_weight_grad=True"),
        (fake_3_none, True, "compute_router_weight_grad=True"),
        (fake_3, False, "compute_router_weight_grad=False"),
    ):
        install(fake)
        with pytest.raises(RuntimeError, match=error):
            call(flag)

    # Positive controls: correct-arity returns pass through the strict unpack
    # with the full tuple intact — values included, not just arity and dtype.
    install(fake_3)
    got = call(True)
    assert torch.equal(got[0], dx) and torch.equal(got[1], postact)
    assert got[2] is not None and got[2].dtype == torch.float32
    assert torch.equal(got[2], d_score)
    install(fake_2)
    got = call(False)
    assert torch.equal(got[0], dx) and torch.equal(got[1], postact)
    assert got[2] is None
    assert seen_classes and all(cls is GemmMoeSm90 for cls in seen_classes), (
        "the role must call the class entry on itself (kernel_variant taken by "
        f"the class, not passed in); saw {seen_classes}"
    )


def test_moe_zero_contraction_up_bias():
    """D=0 (K=0) is rejected up front, not silently degraded.

    A zero contraction dim has no real workload; the primitive refuses it
    instead of emulating degenerate epilogue math host-side.
    """
    x = torch.zeros(2, 0, dtype=torch.bfloat16)
    gate_up_weight = torch.zeros(1, 4, 0, dtype=torch.bfloat16)
    gate_up_bias = torch.ones(1, 4, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="contraction"):
        moe_up_proj_act(
            x, gate_up_weight, offsets, gate_up_bias=gate_up_bias, tune_mode=TuneMode.STATIC
        )


def test_moe_zero_contraction_down_recompute():
    """D=0 (K=0) is rejected on the down projection as well."""
    preact = torch.ones(2, 4, dtype=torch.bfloat16)
    down_weight = torch.zeros(1, 0, 2, dtype=torch.bfloat16)
    grad_output = torch.zeros(2, 0, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="contraction"):
        moe_down_proj_dact(
            grad_output,
            down_weight,
            preact,
            offsets,
            router_weights=torch.zeros(2),
            compute_router_weight_grad=True,
            tune_mode=TuneMode.STATIC,
        )


def test_moe_expert_offsets_host_contract(monkeypatch):
    """CPU expert_offsets get content checks before any backend call.

    Legal ``[0, 2, 4]`` offsets reach the backend exactly once (offsets and
    reference outputs pass through untouched); each illegal variant — nonzero
    start, negative start, wrong last entry, non-monotone — raises ValueError
    with the backend called zero times. The spy patches the class entry the
    role method calls; the recorded receiver class and kwargs pin the MoE call
    shape (dynamic scheduler, gather off, no kernel_variant passed in: the
    class entry takes the token itself).
    """
    torch.manual_seed(29)
    tokens, hidden, intermediate, experts = 4, 16, 8, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    ref_preact, ref_postact = _up_reference(
        x, gate_up_weight, None, offsets, torch.arange(tokens), "swiglu"
    )
    ref_preact = ref_preact.to(torch.bfloat16)
    ref_postact = ref_postact.to(torch.bfloat16)

    calls = []

    def spy(cls, a, b, **kwargs):
        calls.append((cls, kwargs))
        return ref_preact.clone(), ref_postact.clone()

    monkeypatch.setattr(GemmBase, "gemm_with_activation", classmethod(spy), raising=False)
    postact, preact = moe_up_proj_act(
        x, gate_up_weight, offsets, activation="swiglu", tune_mode=TuneMode.STATIC
    )
    assert len(calls) == 1
    cls, kwargs = calls[0]
    assert cls is GemmMoeSm90, "the role must call the class entry on its own class"
    assert kwargs["cu_seqlens_m"] is offsets
    assert kwargs["A_idx"] is None
    assert kwargs["dynamic_scheduler"] is True
    assert "kernel_variant" not in kwargs, "MoE never passes the token; the class entry takes it"
    assert torch.equal(preact, ref_preact) and torch.equal(postact, ref_postact)

    for bad in ([1, 2, 4], [-1, 2, 4], [0, 2, 3], [2, 0, 2]):
        with pytest.raises(ValueError):
            moe_up_proj_act(
                x,
                gate_up_weight,
                torch.tensor(bad, dtype=torch.int32),
                activation="swiglu",
                tune_mode=TuneMode.STATIC,
            )
    assert len(calls) == 1  # no illegal input reached the backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("activation", "layout", "dtype"),
    [
        ("swiglu", "concat", torch.bfloat16),
        ("swiglu", "interleaved", torch.bfloat16),
        ("swiglu", "concat", torch.float16),
        ("geglu", "concat", torch.bfloat16),
        ("reglu", "concat", torch.bfloat16),
        ("silu", "concat", torch.bfloat16),
        ("relu", "concat", torch.bfloat16),
        ("relu_sq", "concat", torch.bfloat16),
        ("gelu_tanh_approx", "concat", torch.bfloat16),
    ],
)
def test_moe_role_shared_mainloop_reference(activation, layout, dtype):
    """Every public activation surface vs the fp32 oracle.

    The role shares the generic SM90 plain-16 mainloop, so each of the seven
    activations must match on preact, postact, dH, A-prime, and dS — one
    representative per surface: bf16 everywhere, the interleaved W13 order on
    the gated flagship (swiglu), fp16 on the same. Gather mode with repeated,
    non-monotonic token rows and an empty middle expert.
    """
    _require_sm90()
    _run_moe_up_down(
        activation,
        layout,
        dtype,
        tokens=4,
        hidden=16,
        intermediate=8,
        experts=3,
        offsets=[0, 2, 2, 4],
        token_indices=[3, 0, 2, 3],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_route_edges():
    """Routing-shape edges: R>T via repeated gather, empty head/tail experts, packed rows."""
    _require_sm90()
    # Routes outnumber tokens (5 routes over 2 tokens, rows repeated).
    _run_moe_up_down(
        "swiglu",
        "concat",
        torch.bfloat16,
        tokens=2,
        hidden=16,
        intermediate=8,
        experts=2,
        offsets=[0, 3, 5],
        token_indices=[1, 1, 0, 0, 1],
        seed=31,
    )
    # First and last experts empty; all rows land in the middle expert.
    _run_moe_up_down(
        "swiglu",
        "concat",
        torch.bfloat16,
        tokens=2,
        hidden=16,
        intermediate=8,
        experts=3,
        offsets=[0, 0, 4, 4],
        token_indices=[1, 0, 0, 1],
        seed=32,
    )
    # Packed expert-major activation rows (token_indices=None).
    _run_moe_up_down(
        "swiglu",
        "concat",
        torch.bfloat16,
        tokens=4,
        hidden=16,
        intermediate=8,
        experts=3,
        offsets=[0, 2, 2, 4],
        token_indices=None,
        seed=33,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_bias_and_preact_modes():
    """Bias/preact plumbing edges: bias-free, fp32 wide bias, store_preact=False."""
    _require_sm90()
    common = {
        "tokens": 4,
        "hidden": 16,
        "intermediate": 8,
        "experts": 3,
        "offsets": [0, 2, 2, 4],
        "token_indices": [3, 0, 2, 3],
    }
    # No bias: the epilogue's bias-free variant.
    _run_moe_up_down("swiglu", "concat", torch.bfloat16, bias_dtype=None, seed=34, **common)
    # fp32 bias on 16-bit activations: the wide-bias broadcast port with the
    # concat relabel applied there (not the materialized 16-bit interleave).
    _run_moe_up_down(
        "swiglu", "concat", torch.bfloat16, bias_dtype=torch.float32, seed=35, **common
    )
    # store_preact=False: preact comes back None, postact is still exact, and
    # the down projection runs from the dtype-cast oracle preact.
    _run_moe_up_down("swiglu", "concat", torch.bfloat16, store_preact=False, seed=36, **common)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_multi_tile_shapes():
    """Tile-crossing shapes: K spanning >=3 k-tiles, and dS summing >=2 output tiles.

    SM90 plain-16 tile_k is 64 (GemmSm90._setup_tiled_mma): hidden=208 spans
    4 k-tiles in both projections. The default tune_mode=TuneMode.STATIC config is (128,
    192) pingpong, so intermediate=256 makes the down GEMM's N=256 span 2
    output tiles (2I = 512 spans 3), exercising the dS reduce sink's
    cross-tile accumulation.
    """
    _require_sm90()
    _run_moe_up_down(
        "swiglu",
        "concat",
        torch.bfloat16,
        tokens=4,
        hidden=208,
        intermediate=8,
        experts=3,
        offsets=[0, 2, 2, 4],
        token_indices=[3, 0, 2, 3],
        seed=37,
    )
    preact, _, d_h, weighted_postact, d_score = _run_moe_up_down(
        "swiglu",
        "concat",
        torch.bfloat16,
        tokens=4,
        hidden=16,
        intermediate=256,
        experts=3,
        offsets=[0, 2, 2, 4],
        token_indices=[3, 0, 2, 3],
        seed=38,
    )
    assert preact.shape == (4, 512) and d_h.shape == (4, 512)
    assert weighted_postact.shape == (4, 256) and d_score.shape == (4,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_tuned_reference(tmp_path, monkeypatch):
    """tune_mode=TuneMode.TUNED (the production path) stays correct and shares one tuning bucket.

    R=6 and R=7 over 3 experts both round up to the _MIN_AVERAGE_ROWS = 32
    bucket floor (cute_gemm.varlen_selector.average_rows_bucket) with every
    other tune-key component held fixed, so the second call must replay the
    first's autotuned winner instead of retuning — pinned empirically by the
    in-memory autotuner cache count.
    """
    # TUNED sweeps persist winners; keep them inside pytest's tmp_path.
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    device = _require_sm90()
    from cute_gemm.gemm_runtime import autotune as epi_autotune
    from cute_gemm.varlen_selector import average_rows_bucket

    assert average_rows_bucket(6, 3) == average_rows_bucket(7, 3) == 32

    def tuner_keys():
        # All in-memory autotune winners (a disk-restored winner counts too).
        return {key for tuner in epi_autotune._MOD_TUNERS.values() for key in tuner.cache}

    for activation in ("swiglu", "silu"):
        results = []
        for route_count, offsets in ((6, [0, 2, 4, 6]), (7, [0, 2, 4, 7])):
            torch.manual_seed(46)
            tokens, hidden, intermediate, experts = 8, 16, 8, 3
            width = 2 * intermediate if activation in _GATED_ACTIVATIONS else intermediate
            dtype = torch.bfloat16
            x = torch.randn(tokens, hidden, device=device, dtype=dtype)
            gate_up_weight = torch.randn(experts, width, hidden, device=device, dtype=dtype)
            gate_up_bias = torch.randn(experts, width, device=device, dtype=dtype)
            offsets_t = torch.tensor(offsets, device=device, dtype=torch.int32)
            token_indices = torch.randint(tokens, (route_count,), device=device, dtype=torch.int32)
            postact, preact = moe_up_proj_act(
                x,
                gate_up_weight,
                offsets_t,
                activation=activation,
                token_indices=token_indices,
                gate_up_bias=gate_up_bias,
                tune_mode=TuneMode.TUNED,
            )
            expected_preact, expected_postact = _up_reference(
                x, gate_up_weight, gate_up_bias, offsets_t, token_indices, activation
            )
            tag = f"{activation}/R={route_count}"
            _assert_outputs((preact, postact), (expected_preact, expected_postact), tag, dtype)
            results.append((preact, postact))
            if route_count == 6:
                # The bucket entry exists (tuned now or restored from the
                # persistent winner cache) and carries the shared bucket.
                keys_after_first = tuner_keys()
                assert any(("avg_expert_m", 32) in key for key in keys_after_first), (
                    f"{activation}: no tuned winner for the shared avg_expert_m=32 bucket"
                )
        assert tuner_keys() == keys_after_first, (
            f"{activation}: R=7 tuned a new key instead of sharing the R=6 bucket"
        )
        assert results[0][0].shape != results[1][0].shape  # distinct R, one bucket winner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("pingpong", [True, False])
@pytest.mark.parametrize("hidden", [64, 72])
def test_moe_role_config_pinned_pingpong_k_boundary(hidden, pingpong):
    """Config-pinned eager validation across warp-specialization and K boundaries.

    INTERNAL validation only (config pinning under torch.compile raises
    NotImplementedError): the class entries with an explicit GemmConfig,
    received by the role and by the generic root. K=64 is exactly one k-tile
    (SM90 plain-16 tile_k = mma_inst_shape_k * 4 = 64) and K=72 adds the
    smallest 8-element tail that keeps 16-byte TMA row alignment; both configs
    are gather-A legal on SM90 (cluster_n == 1, tile_n != 208, static
    persistence) and within the pingpong tile limits. The activation axis is
    pinned by shared_mainloop; swiglu stands in here.
    """
    device = _require_sm90()
    from cute_gemm.gemm_config import GemmConfig

    activation = "swiglu"
    config = GemmConfig(
        tile_m=128,
        tile_n=128,
        cluster_m=2 if pingpong else 1,
        cluster_n=1,
        pingpong=pingpong,
        swap_ab=False,
        device_capacity=9,
        is_dynamic_persistent=False,
    )
    torch.manual_seed(47)
    tokens, intermediate, experts = 4, 32, 2
    gated = activation in _GATED_ACTIVATIONS
    width = 2 * intermediate if gated else intermediate
    dtype = torch.bfloat16
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    gate_up_weight = torch.randn(experts, width, hidden, device=device, dtype=dtype)
    gate_up_bias = torch.randn(experts, width, device=device, dtype=dtype)
    down_weight = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype)
    grad_output = torch.randn(tokens, hidden, device=device, dtype=dtype)
    offsets = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    token_indices = torch.tensor([3, 1, 2, 0], device=device, dtype=torch.int32)
    scores = torch.tensor([0.0, 0.5, -1.0, 2.0], device=device, dtype=torch.float32)
    expected_preact, expected_postact = _up_reference(
        x, gate_up_weight, gate_up_bias, offsets, token_indices, activation
    )

    # The role entry (GemmMoeSm90, which owns the kernel_variant token) and,
    # with the same config, the generic dispatch (no role): both run the
    # identical inherited mainloop and must agree with the same oracle.
    for base, arm in ((GemmMoeSm90, "role"), (GemmBase, "generic")):
        tag = f"{activation}/K={hidden}/pingpong={pingpong}/{arm}"
        preact, postact = base.gemm_with_activation(
            x,
            gate_up_weight.mT,
            bias=gate_up_bias,
            activation=activation,
            cu_seqlens_m=offsets,
            A_idx=token_indices,
            dynamic_scheduler=True,
            tune_mode=TuneMode.STATIC,
            config=config,
            concat_layout=("B", "bias") if gated else None,
        )
        d_h, weighted_postact, d_score = base.gemm_with_activation_grad(
            grad_output,
            down_weight,
            preact,
            activation=activation,
            colvec_scale=scores,
            colvec_reduce=True,
            cu_seqlens_m=offsets,
            A_idx=token_indices,
            dynamic_scheduler=True,
            tune_mode=TuneMode.STATIC,
            config=config,
        )
        expected_dh, expected_a, expected_ds = _down_reference(
            grad_output, down_weight, preact, offsets, token_indices, scores, activation
        )
        _assert_outputs(
            (preact, postact, d_h, weighted_postact, d_score),
            (expected_preact, expected_postact, expected_dh, expected_a, expected_ds),
            tag,
            dtype,
        )


def test_moe_role_codec_and_ctor_contract():
    """Variant-token plumbing and the restricted construction contract, CPU-side.

    The class-owned variant_id is the wire token that crosses the
    torch.compile boundary (kernel_variant); registry resolution and the
    duplicate/unknown-token rules are pinned by
    test_kernel_variant_registry_contract below. The role stays a direct
    subclass of the generic SM90 base with its guarded constructor: plain
    16-bit WGMMA only, serial split-K by default.
    """
    import cutlass

    from cute_gemm.gemm_config import SplitKMode
    from cute_gemm.gemm_runtime.identity import resolve_gemm_class, static_gemm_class_ref
    from cute_gemm.gemm_sm90 import GemmSm90

    # Token ownership lives on the class; the generic base declares none.
    assert GemmSm90.variant_id is None
    assert GemmMoeSm90.variant_id == "moe_sm90"
    with pytest.raises(TypeError, match="variant_id"):
        type("InvalidTokenRole", (GemmSm90,), {"variant_id": 123}).get_kernel_variant()

    # The async-compile worker resolution path (first statement of
    # gemm_runtime.host._compile_gemm_epi) still resolves the class by ref.
    assert resolve_gemm_class(static_gemm_class_ref(GemmMoeSm90)) is GemmMoeSm90
    assert issubclass(GemmMoeSm90, GemmSm90)

    with pytest.raises(ValueError, match="plain BF16/FP16"):
        GemmMoeSm90(cutlass.Float32, cutlass.Float32, (128, 128), (1, 1, 1))
    with pytest.raises(ValueError, match="MoE contract"):
        GemmMoeSm90(cutlass.Float16, cutlass.Float16, (128, 128), (1, 1, 1), mma_is_rs=True)
    with pytest.raises(ValueError, match="MoE contract"):
        GemmMoeSm90(cutlass.Float16, cutlass.Float16, (128, 128), (1, 1, 1), transform_a=object())
    role = GemmMoeSm90(cutlass.Float16, cutlass.Float16, (128, 128), (2, 1, 1))
    assert role.split_k_mode == SplitKMode.SERIAL == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_minted_mro_order(monkeypatch, tmp_path):
    """The MoE role is live on both execution paths and in mod identity.

    The role's identity is asserted at the wire — the decode spy sees the
    role string cross the custom-op boundary on every execution, warm or
    cold — and in the minted MRO (GemmMoeSm90 before GemmSm90: the role
    survived the kernel_variant round-trip, not the generic map). A cold
    eager plan (fresh metadata, fresh cache dir so the jit cache cannot
    shortcut the construction) instantiates the role class itself, and the
    role separates the mod digest (an unpinned mod would silently share
    kernels with the generic class).
    """
    device = _require_sm90()
    import cute_gemm.cache
    import cute_gemm.epilogue.frontend as frontend_module
    from cute_gemm.epilogue.library import linear_act_mod
    from cute_gemm.gemm_sm90 import GemmSm90

    torch.manual_seed(48)
    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(3, 16, 16, device=device, dtype=torch.bfloat16)
    gate_up_bias = torch.randn(3, 16, device=device, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 2, 4], device=device, dtype=torch.int32)

    def up(x, w13, b13, offs):
        return moe_up_proj_act(
            x, w13, offs, activation="swiglu", gate_up_bias=b13, tune_mode=TuneMode.STATIC
        )

    variants = _record_kernel_variants(monkeypatch)
    torch.compile(up, fullgraph=True)(x, gate_up_weight, gate_up_bias, offsets)
    assert "moe_sm90" in variants, (
        "the compiled call did not carry the MoE role across the op boundary"
    )
    minted = [
        cls
        for cls in vars(frontend_module).values()
        if isinstance(cls, type) and GemmMoeSm90 in cls.__mro__
    ]
    assert minted, "no minted frontend class carries the MoE role in its MRO"
    for cls in minted:
        mro = cls.__mro__
        assert mro.index(GemmMoeSm90) < mro.index(GemmSm90), (
            "the role must precede the generic base in the minted MRO"
        )

    # Cold eager plan (fresh metadata: no bias) must instantiate the role class.
    calls = []
    original_init = GemmMoeSm90.__init__

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(cute_gemm.cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(GemmMoeSm90, "__init__", spy)
    moe_up_proj_act(
        torch.randn(6, 16, device=device, dtype=torch.bfloat16),
        torch.randn(2, 16, 16, device=device, dtype=torch.bfloat16),
        torch.tensor([0, 4, 6], device=device, dtype=torch.int32),
        activation="swiglu",
        tune_mode=TuneMode.STATIC,
    )
    assert calls, "cold eager MoE call did not instantiate GemmMoeSm90"

    plain = linear_act_mod("swiglu", gated=True, has_c=False, has_rowvec=True, has_colvec=False)
    role = GemmMoeSm90.bind_epilogue(plain)
    assert plain.semantic_digest != role.semantic_digest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_cold_subprocess_identity(tmp_path):
    """A cold process (fresh cache dir) resolves the role token and computes.

    The decode path is exactly what an async-compile worker or a fresh
    interpreter uses: the role module registers "moe_sm90" into the neutral
    registry at import, no other in-process state. One tiny eager call plus a
    numeric check proves the resolved class plans, compiles, and launches.
    """
    _require_sm90()
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import torch
        import torch.nn.functional as F
        from cute_gemm import TuneMode, moe_up_proj_act
        from cute_gemm.fused_grouped_gemm.gemm_sm90 import GemmMoeSm90
        from cute_gemm.gemm_base import GemmBase
        from cute_gemm.gemm_runtime.identity import KERNEL_VARIANT_CLASSES

        cls = KERNEL_VARIANT_CLASSES["moe_sm90"]
        assert cls is GemmMoeSm90, (cls, GemmMoeSm90)
        assert GemmBase.resolve_kernel_variant("moe_sm90") is GemmMoeSm90

        device = "cuda"
        torch.manual_seed(49)
        x = torch.randn(2, 16, device=device, dtype=torch.bfloat16)
        gate_up_weight = torch.randn(1, 16, 16, device=device, dtype=torch.bfloat16)
        gate_up_bias = torch.randn(1, 16, device=device, dtype=torch.bfloat16)
        offsets = torch.tensor([0, 2], device=device, dtype=torch.int32)
        postact, preact = moe_up_proj_act(
            x, gate_up_weight, offsets, activation="swiglu", gate_up_bias=gate_up_bias,
            tune_mode=TuneMode.STATIC,
        )
        assert preact.shape == (2, 16) and postact.shape == (2, 8)
        values = F.linear(x.float(), gate_up_weight[0].float()) + gate_up_bias[0].float()
        expected = F.silu(values[:, :8]) * values[:, 8:]
        torch.testing.assert_close(postact.float(), expected, atol=0.05, rtol=0.03)
        print("COLD_MOE_SUBPROCESS_OK")
        """
    )
    env = dict(os.environ, DLKERNEL_CACHE_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the assertion below reports stdout/stderr on failure
    )
    assert result.returncode == 0, f"cold subprocess failed:\n{result.stdout}\n{result.stderr}"
    assert "COLD_MOE_SUBPROCESS_OK" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_role_async_compile_worker(tmp_path, monkeypatch):
    """A cold MoE-role compile resolves the class inside the async-compile pool.

    A fresh CACHE_DIR forces the jit miss; the pool worker resolves the
    STATIC GemmClassRef by module:qualname before tracing, so a worker-side
    failure would surface as an in-process fallback warning. The
    geglu/interleaved/bias-free combination is unique to this test: an
    earlier in-process compile of the same key would satisfy the in-memory
    cache and the pool would never be engaged.
    """
    import time
    import warnings

    import cute_gemm.cache as cache_state
    import cute_gemm.epilogue.frontend as frontend_module
    from cute_gemm.cache import async_compile

    device = _require_sm90()
    torch.manual_seed(50)
    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(3, 16, 16, device=device, dtype=torch.bfloat16)
    # The interleaved layout contract puts the [g0, u0, g1, ...] reordering on
    # the caller; the oracle below consumes the concat-space weight.
    kernel_weight = _interleave_rows(gate_up_weight, 3, 8)
    offsets = torch.tensor([0, 2, 2, 4], device=device, dtype=torch.int32)

    monkeypatch.setattr(cache_state, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_state, "CACHE_ENABLED", True)
    _purge_role_minted_classes()
    with async_compile.pool_scope() as pool:
        submitted_before = pool.n_submitted
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deadline = time.monotonic() + 600
            while True:
                try:
                    postact, preact = moe_up_proj_act(
                        x,
                        kernel_weight,
                        offsets,
                        activation="geglu",
                        gate_up_layout="interleaved",
                        tune_mode=TuneMode.STATIC,
                    )
                    break
                except async_compile.CompilePending:
                    assert time.monotonic() < deadline, "MoE pool compile never completed"
                    time.sleep(0.2)
        fallbacks = [str(w.message) for w in caught if "async compile failed" in str(w.message)]
        assert not fallbacks, f"pool worker failed, compiled in-process instead: {fallbacks}"
        assert pool.n_submitted - submitted_before >= 1, (
            "the cold MoE-role compile never reached the pool (in-memory or disk hit)"
        )
    # The class the worker had to resolve is the role, and the post-pool mint
    # still carries it ahead of the generic base.
    from cute_gemm.gemm_sm90 import GemmSm90

    minted = [
        cls
        for cls in vars(frontend_module).values()
        if isinstance(cls, type) and GemmMoeSm90 in cls.__mro__
    ]
    assert minted, "no minted frontend class carries the MoE role after the pool compile"
    for cls in minted:
        assert cls.__mro__.index(GemmMoeSm90) < cls.__mro__.index(GemmSm90)
    # Numerics: the oracle consumes the concat-space weight; the kernel got
    # the same values reordered to the interleaved [g0, u0, g1, ...] rows.
    expected_preact, expected_postact = _up_reference(
        x,
        gate_up_weight,
        None,
        offsets,
        torch.arange(4, device=device),
        "geglu",
    )
    _assert_close_scaled(preact, expected_preact, "async geglu preactivation", torch.bfloat16)
    _assert_close_scaled(postact, expected_postact, "async geglu postactivation", torch.bfloat16)


# Per-rank production shapes sampled from .vscode/moe_shape.md (EP=8: E_local
# = E/8; H, I, topk per model). The balanced matrix below sweeps the smallest
# and the largest-hidden shape at the minimal (t=1) and steady-state (t=4001)
# token tiers; the other two shapes survive only as switch-case hosts (packed
# no-score on qwen35_122b, the 768-local-expert extreme on metamoe). Balanced
# routing with a repeated-gather index, concat W13 + bias, fp32 scores,
# reduce on.
_PRODUCTION_SHAPES = {
    "qwen35_35b": (2048, 512, 32, 8),
    "qwen35_122b": (3072, 1024, 32, 8),
    "dsv32": (7168, 2048, 32, 8),
    "metamoe": (2048, 768, 768, 8),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", (1, 4001))
@pytest.mark.parametrize("model", ["qwen35_35b", "dsv32"])
def test_moe_sm90_production_shapes_match_reference(model, tokens):
    """moe_shape.md model shapes hold the same math as the tiny fixtures.

    The minimal (t=1) and steady-state (t=4001) token tiers on the smallest
    and the largest-hidden shape, run through the switch engine at its
    production defaults; the tiers and shapes between proved same-path
    repeats of these.
    """
    _production_switch_case({"name": f"{model}_t{tokens}", "model": model, "tokens": tokens})


def _production_switch_case(case):
    """One adversarial switch combination at a production shape.

    ``case`` mixes the primitive switches a balanced matrix cannot: W13
    layout, bias width (absent / 16-bit / fp32), score regime, routing skew,
    gather (repeated, non-monotonic rows) vs packed input, reduce on/off, and
    dtype, against a moe_shape.md model. Keys omitted by ``case`` fall back
    to the production defaults — the balanced-matrix configuration below.
    Each kept case pins one unique combination — interleaved+fp32-bias+
    signed scores, packed+no-score+no-reduce, hot routing, fp16 — the
    dropped ones were recombinations of independently covered paths.
    """
    device = _require_sm90()
    case = {
        "activation": "swiglu",
        "layout": "concat",
        "bias": torch.bfloat16,
        "scores": "positive",
        "routing": "balanced",
        "gather": True,
        "dtype": torch.bfloat16,
        "reduce": True,
        "seed": sum(map(ord, case["name"])) % 100000,
        **case,
    }
    name = case["name"]
    model, tokens = case["model"], case["tokens"]
    activation = case["activation"]
    layout = case["layout"]
    bias_dtype = case["bias"]
    score_mode = case["scores"]
    hidden, intermediate, experts, topk = _PRODUCTION_SHAPES[model]
    gated = activation in _GATED_ACTIVATIONS
    width = 2 * intermediate if gated else intermediate
    dtype = case["dtype"]
    torch.manual_seed(case["seed"])

    if case["routing"] == "hot":
        # One 70% hot expert against a balanced remainder.
        hot = int(tokens * topk * 0.70)
        rest, rem = divmod(tokens * topk - hot, experts - 1)
        counts = torch.full((experts,), rest, dtype=torch.int64)
        counts[0] = hot
        counts[1 : 1 + rem] += 1
    else:
        counts = _balanced_counts(tokens * topk, experts)
    routes = int(counts.sum())
    offsets = _offsets_from_counts(counts, device)

    token_indices = None
    if case["gather"]:
        gen = torch.Generator(device=device).manual_seed(41 + tokens)
        token_indices = (
            torch.randint(tokens, (routes,), generator=gen, device=device, dtype=torch.int32)
            if tokens > 1
            else torch.zeros(routes, device=device, dtype=torch.int32)
        )

    if score_mode == "signed":
        scores = torch.linspace(-1.0, 1.0, routes, device=device, dtype=torch.float32)
        scores[::5] = 0.0
    elif score_mode == "none":
        scores = None
    else:  # positive ramp
        scores = torch.linspace(0.125, 1.0, routes, device=device, dtype=torch.float32)

    # Gather mode reads x[token_indices[r]] with T source rows; packed mode
    # (token_indices=None) requires the activation ALREADY expert-major
    # packed, so its row count is R = offsets[-1], not T.
    x_rows = routes if token_indices is None else tokens
    x = (0.02 * torch.randn(x_rows, hidden, device=device, dtype=dtype)).contiguous()
    gate_up_weight = (
        0.02 * torch.randn(experts, width, hidden, device=device, dtype=dtype)
    ).contiguous()
    interleave = layout == "interleaved" and gated
    kernel_weight = (
        _interleave_rows(gate_up_weight, experts, intermediate).contiguous()
        if interleave
        else gate_up_weight
    )
    if bias_dtype is None:
        kernel_bias = ref_bias = None
    else:
        ref_bias = 0.02 * torch.randn(experts, width, device=device, dtype=bias_dtype)
        kernel_bias = _interleave_rows(ref_bias, experts, intermediate) if interleave else ref_bias
    down_weight = (
        0.02 * torch.randn(experts, hidden, intermediate, device=device, dtype=dtype)
    ).contiguous()
    grad_output = (0.02 * torch.randn(x_rows, hidden, device=device, dtype=dtype)).contiguous()
    gather_indices = (
        token_indices if token_indices is not None else torch.arange(routes, device=device)
    )

    postact, preact = moe_up_proj_act(
        x,
        kernel_weight,
        offsets,
        activation=activation,
        gate_up_layout=layout if gated else "concat",
        token_indices=token_indices,
        gate_up_bias=kernel_bias,
        store_preact=True,
        tune_mode=TuneMode.STATIC,
    )
    expected_preact, expected_postact = _up_reference(
        x, gate_up_weight, ref_bias, offsets, gather_indices, activation
    )
    d_h, weighted_postact, d_score = moe_down_proj_dact(
        grad_output,
        down_weight,
        preact,
        offsets,
        activation=activation,
        token_indices=token_indices,
        router_weights=scores,
        compute_router_weight_grad=case["reduce"],
        tune_mode=TuneMode.STATIC,
    )
    expected_dh, expected_a, expected_ds = _down_reference(
        grad_output, down_weight, expected_preact, offsets, gather_indices, scores, activation
    )
    _assert_outputs(
        (preact, postact, d_h, weighted_postact, d_score),
        (expected_preact, expected_postact, expected_dh, expected_a, expected_ds),
        f"{name}/{dtype}",
        dtype,
    )


# Deviations from the engine defaults only (swiglu / concat / bf16 bias /
# positive scores / balanced routing / gather / bf16 / reduce on).
_PRODUCTION_SWITCH_CASES = [
    {
        "name": "interleaved_fp32bias_signed",
        "model": "qwen35_35b",
        "tokens": 123,
        "layout": "interleaved",
        "bias": torch.float32,
        "scores": "signed",
    },
    {
        "name": "silu_packed_noscore",
        "model": "qwen35_122b",
        "tokens": 4001,
        "activation": "silu",
        "gather": False,
        "scores": "none",
        "reduce": False,
    },
    {
        "name": "swiglu_hot_routing",
        "model": "metamoe",
        "tokens": 4001,
        "routing": "hot",
    },
    {
        "name": "swiglu_fp16_production",
        "model": "qwen35_35b",
        "tokens": 4001,
        "bias": torch.float16,
        "dtype": torch.float16,
    },
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "case", _PRODUCTION_SWITCH_CASES, ids=[c["name"] for c in _PRODUCTION_SWITCH_CASES]
)
def test_moe_sm90_production_switch_matrix(case):
    """Adversarial switch combinations at production shapes (moe_shape.md)."""
    _production_switch_case(case)


# ---------------------------------------------------------------------------
# Host-side class selection and the neutral kernel_variant registry
# ---------------------------------------------------------------------------


def test_gemm_class_select_matrix():
    """GemmBase.select_gemm_class maps each capability major to a generic base.

    8 -> GemmSm80, 9 -> GemmSm90, 10 and 11 -> GemmSm100, 12 -> GemmSm120.
    The declarations are pinned alongside: SM100 must include major 11
    (B300 reports capability 11 while arch stays 100), and SM120 must
    override the (9,) it would inherit from GemmSm90.
    """
    from cute_gemm.gemm_sm80 import GemmSm80
    from cute_gemm.gemm_sm90 import GemmSm90
    from cute_gemm.gemm_sm100 import GemmSm100
    from cute_gemm.gemm_sm120 import GemmSm120

    assert GemmBase.select_gemm_class(8) is GemmSm80
    assert GemmBase.select_gemm_class(9) is GemmSm90
    assert GemmBase.select_gemm_class(10) is GemmSm100
    assert GemmBase.select_gemm_class(11) is GemmSm100
    assert GemmBase.select_gemm_class(12) is GemmSm120
    # The concrete receiver returns itself, SM80 included (the placeholder
    # class must still hold the class-selection contract).
    assert GemmSm80.select_gemm_class(8) is GemmSm80

    assert GemmSm80.supported_sm_majors == (8,)
    assert GemmSm90.supported_sm_majors == (9,)
    assert GemmSm100.supported_sm_majors == (10, 11)
    assert GemmSm120.supported_sm_majors == (12,)
    # The MoE role makes no declaration of its own: it inherits SM90's set.
    assert GemmMoeSm90.supported_sm_majors == (9,)


def test_gemm_class_select_concrete_and_errors():
    """Concrete classes return themselves; wrong/unknown majors raise by category.

    A concrete class applies exactly on its declared majors and returns cls
    itself (a class, never an instance). A known major outside the set is a
    TypeError (the class exists but does not apply) naming the refusing
    class; an unknown major is a ValueError on the generic root and on a
    concrete receiver alike — the categories the old EpiMod kernel-base
    selection established. The SM120 counterexample: GemmSm120 subclasses
    GemmSm90 for mainloop reuse, yet must never be selected for major 9, nor
    GemmSm90 for 12. The MoE role never dispatches generically for its own
    arch: on SM90 it selects itself (the minted class keeps the MoE mainloop).
    """
    from cute_gemm.gemm_sm90 import GemmSm90
    from cute_gemm.gemm_sm100 import GemmSm100
    from cute_gemm.gemm_sm120 import GemmSm120

    assert GemmSm90.select_gemm_class(9) is GemmSm90
    assert GemmSm100.select_gemm_class(10) is GemmSm100
    assert GemmSm100.select_gemm_class(11) is GemmSm100
    assert GemmSm120.select_gemm_class(12) is GemmSm120
    assert GemmMoeSm90.select_gemm_class(9) is GemmMoeSm90

    with pytest.raises(TypeError):
        GemmSm90.select_gemm_class(10)
    with pytest.raises(TypeError) as exc_info:
        GemmMoeSm90.select_gemm_class(10)
    message = str(exc_info.value)
    assert "GemmMoeSm90" in message and "10" in message
    assert issubclass(GemmSm120, GemmSm90)  # the inheritance counterexample
    with pytest.raises(TypeError):
        GemmSm120.select_gemm_class(9)
    with pytest.raises(TypeError):
        GemmSm90.select_gemm_class(12)

    with pytest.raises(ValueError, match="unsupported SM major"):
        GemmBase.select_gemm_class(13)
    with pytest.raises(ValueError, match="unsupported SM major"):
        GemmMoeSm90.select_gemm_class(13)


def test_kernel_variant_registry_contract(monkeypatch):
    """The neutral token registry: declare, resolve, duplicate, unknown, minted.

    Built-in roles register at their module's import. resolve_kernel_variant
    (None) constant-returns GemmBase without touching the registry; unknown
    tokens raise instead of falling back generic; a taken token refuses a
    different class while staying idempotent for its own; type()-minted
    classes never enter the table (no auto-registration).
    """
    import cute_gemm.gemm_runtime.identity as identity_module
    from cute_gemm.gemm_sm90 import GemmSm90

    # Registered at import of the role module; values are classes, not names.
    assert identity_module.KERNEL_VARIANT_CLASSES.get("moe_sm90") is GemmMoeSm90
    assert all(isinstance(v, type) for v in identity_module.KERNEL_VARIANT_CLASSES.values())
    assert all(issubclass(v, GemmBase) for v in identity_module.KERNEL_VARIANT_CLASSES.values())

    # Declaration read: declared token or None; no registry involved.
    assert GemmMoeSm90.get_kernel_variant() == "moe_sm90"
    assert GemmBase.get_kernel_variant() is None
    assert GemmSm90.get_kernel_variant() is None

    # Resolution: token -> class; None -> the generic root, identically every
    # time and from any receiver. An exploding resolver AND an exploding
    # registry mapping prove the None path consults neither — a direct dict
    # read (in / get / []) would trip the mapping just as a resolver call
    # trips the explode spy.
    def explode(token):
        raise AssertionError("registry consulted for the generic None token")

    class _ExplodingRegistry:
        def __contains__(self, token):
            raise AssertionError(f"registry mapping read for the None token ({token!r})")

        def get(self, token, default=None):
            raise AssertionError(f"registry mapping read for the None token ({token!r})")

        def __getitem__(self, token):
            raise AssertionError(f"registry mapping read for the None token ({token!r})")

    monkeypatch.setattr(identity_module, "resolve_kernel_variant_class", explode)
    monkeypatch.setattr(identity_module, "KERNEL_VARIANT_CLASSES", _ExplodingRegistry())
    assert GemmBase.resolve_kernel_variant(None) is GemmBase
    assert GemmMoeSm90.resolve_kernel_variant(None) is GemmBase
    monkeypatch.undo()
    assert GemmBase.resolve_kernel_variant("moe_sm90") is GemmMoeSm90

    # Unknown token: fail closed, never a generic fallback.
    with pytest.raises(ValueError, match="unknown kernel_variant"):
        GemmBase.resolve_kernel_variant("moe_sm100")

    # Same token, same class: idempotent re-registration.
    GemmMoeSm90.register_kernel_variant()
    assert identity_module.KERNEL_VARIANT_CLASSES["moe_sm90"] is GemmMoeSm90

    # Same token, different class: refused, table unchanged.
    squatter = type("SquatterRole", (GemmSm90,), {"variant_id": "moe_sm90"})
    with pytest.raises(ValueError, match="moe_sm90"):
        squatter.register_kernel_variant()
    assert identity_module.KERNEL_VARIANT_CLASSES["moe_sm90"] is GemmMoeSm90

    # A class without its own token cannot register at all.
    tokenless = type("TokenlessRole", (GemmSm90,), {"variant_id": None})
    with pytest.raises(TypeError, match="variant_id"):
        tokenless.register_kernel_variant()

    # Minted dynamic classes inherit the role token but never auto-register.
    minted = type("MintedMod", (GemmMoeSm90,), {})
    assert minted.variant_id == "moe_sm90"
    assert minted not in identity_module.KERNEL_VARIANT_CLASSES.values()


def test_identity_variant_registry_keeps_identity_a_leaf():
    """The token registry adds no module-level cute_gemm import to identity.

    identity hosts the registry precisely because it is a leaf module: its
    import must not pull kernel/MoE modules (and through them CUDA) nor know
    GemmBase. Asserted statically on module-level statements, including
    imports hiding inside top-level try blocks and if guards — those execute
    at import time just the same; function bodies import lazily by design.
    """
    import ast
    import pathlib

    import cute_gemm.gemm_runtime.identity as identity_module

    def _statement_imports(stmts):
        found = []
        for stmt in stmts:
            if isinstance(stmt, ast.Import):
                found.extend(alias.name for alias in stmt.names)
            elif isinstance(stmt, ast.ImportFrom):
                found.append(stmt.module or "")
            elif isinstance(stmt, ast.Try):
                found.extend(_statement_imports(stmt.body))
                for handler in stmt.handlers:
                    found.extend(_statement_imports(handler.body))
                found.extend(_statement_imports(stmt.orelse))
                found.extend(_statement_imports(stmt.finalbody))
            elif isinstance(stmt, ast.If):
                found.extend(_statement_imports(stmt.body))
                found.extend(_statement_imports(stmt.orelse))
        return found

    tree = ast.parse(pathlib.Path(identity_module.__file__).read_text())
    imported = _statement_imports(tree.body)
    dlkernel_imports = sorted(n for n in imported if n == "cute_gemm" or n.startswith("cute_gemm."))
    assert not dlkernel_imports, (
        f"identity must stay a leaf; module-level imports: {dlkernel_imports}"
    )
    assert not hasattr(identity_module, "GemmBase")
    # The registry itself lives here: token -> class objects, populated by
    # role modules at their import, never by this module.
    assert isinstance(identity_module.KERNEL_VARIANT_CLASSES, dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_interleaved_calls_and_state_holds_no_tensors():
    """Two MoE up/down shapes alternated A,B,A,B each match their own
    reference and reuse deterministically, and the class-side state added by
    the class-polymorphism refactor (token registry, the math mods'
    ``_bound_epilogues`` memo, host-entry classmethods) retains no input,
    output, or scratch tensor across calls."""
    device = _require_sm90()

    def make_case(tokens, hidden, intermediate, experts, empty_expert, seed):
        torch.manual_seed(seed)
        base, extra = divmod(tokens, experts)
        counts = torch.tensor(
            [0 if g == empty_expert else base + int(g < extra) for g in range(experts)]
        )
        offsets = _offsets_from_counts(counts, device)
        routed = int(offsets[-1])
        return (
            torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16),
            torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16),
            torch.randn(experts, 2 * intermediate, device=device, dtype=torch.bfloat16),
            torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16),
            torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16),
            torch.randn(routed, device=device, dtype=torch.float32),
            torch.randint(0, tokens, (routed,), device=device, dtype=torch.int32),
            offsets,
        )

    def run_case(c):
        x, w13, b13, w3, go, scores, idx, offsets = c
        postact, preact = moe_up_proj_act(
            x,
            w13,
            offsets,
            activation="swiglu",
            gate_up_layout="concat",
            token_indices=idx,
            gate_up_bias=b13,
            store_preact=True,
            tune_mode=TuneMode.STATIC,
        )
        dh, a_prime, ds = moe_down_proj_dact(
            go,
            w3,
            preact,
            offsets,
            activation="swiglu",
            token_indices=idx,
            router_weights=scores,
            compute_router_weight_grad=True,
            tune_mode=TuneMode.STATIC,
        )
        return preact, postact, dh, a_prime, ds

    def check_case(c, got, label):
        x, w13, b13, w3, go, scores, idx, offsets = c
        gather = idx.long()
        e_preact, e_postact = _up_reference(x, w13, b13, offsets, gather, "swiglu")
        e_dh, e_a, e_ds = _down_reference(go, w3, got[0], offsets, gather, scores, "swiglu")
        _assert_outputs(
            got,
            (e_preact, e_postact, e_dh, e_a, e_ds),
            label,
            torch.bfloat16,
            d_score_dtype=torch.float32,
        )

    a = make_case(384, 128, 64, 4, empty_expert=2, seed=101)
    b = make_case(512, 256, 128, 8, empty_expert=5, seed=202)
    a1, b1, a2, b2 = run_case(a), run_case(b), run_case(a), run_case(b)
    for label, case, got in (("A#1", a, a1), ("B#1", b, b1), ("A#2", a, a2), ("B#2", b, b2)):
        check_case(case, got, label)
    # Same metadata -> warm plan reuse must be deterministic, proving the
    # alternation never crossed the two shapes' plans.
    for g1, g2 in zip(a1, a2):
        assert torch.equal(g1, g2), "interleaved A#1/A#2 outputs differ"
    for g1, g2 in zip(b1, b2):
        assert torch.equal(g1, g2), "interleaved B#1/B#2 outputs differ"

    # Memory safety: no class-side/registry/memo state may hold tensors of a
    # finished call (weakrefs must die once the caller drops its references).
    import gc
    import weakref

    c = make_case(256, 128, 64, 4, empty_expert=1, seed=303)
    got = run_case(c)
    tensors = list(c) + list(got)
    refs = [weakref.ref(t) for t in tensors if isinstance(t, torch.Tensor)]
    del c, got, tensors, a1, b1, a2, b2
    gc.collect()
    torch.cuda.synchronize()
    alive = [i for i, r in enumerate(refs) if r() is not None]
    assert not alive, f"class/registry/bound-mod state retains {len(alive)} tensor(s)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_expert_mlp_full_backward_chain(tmp_path, monkeypatch):
    """T6 (plan v4 §5): the composed expert MLP backward — every
    kernel-expressed gradient is derived from the primitives' own outputs and
    compared against an fp32 autograd reference.

    The dact primitive returns d_gate_up (the d_preact) in the interleaved
    [R, 2I] preact space; the caller maps it back to the concat [gate; up]
    storage that moe_up_proj_act consumed, then feeds it everywhere: dx =
    d_gate_up @ w13 per expert segment, dw13 = d_gate_up^T @ x via the
    varlen-K GEMM, db13 = the per-expert segment sum.  dw2 stays dY^T @
    kernel postact.  The wgrad GEMMs run TUNED so the varlen-K sweep path is
    exercised on the way.

    Geometry keeps 2I (768) != hidden (512): no kernel surface shares the dx
    shape, so a misrouted output cannot pass on shape coincidence.  Every
    comparison runs through _assert_close_scaled, whose atol scales with the
    oracle magnitude with no constant floor — an all-zero kernel output
    fails wherever the oracle carries signal.
    """
    # TUNED sweeps persist winners; keep them inside pytest's tmp_path.
    monkeypatch.setenv("DLKERNEL_CACHE_DIR", str(tmp_path))
    from cute_gemm.gemm_interface import gemm as plain_gemm

    if torch.cuda.get_device_capability(0)[0] != 9:
        pytest.skip("SM90 only")
    device = torch.device("cuda")
    torch.manual_seed(11)
    tokens, hidden, intermediate, experts = 96, 512, 384, 4
    dtype = torch.bfloat16
    counts = _balanced_counts(tokens, experts)
    offsets = torch.as_tensor(_offsets_from_counts(counts), device=device, dtype=torch.int32)

    x = 0.05 * torch.randn(tokens, hidden, dtype=dtype, device=device)
    w13 = 0.05 * torch.randn(experts, 2 * intermediate, hidden, dtype=dtype, device=device)
    b13 = 0.05 * torch.randn(experts, 2 * intermediate, dtype=dtype, device=device)
    w2 = 0.05 * torch.randn(experts, hidden, intermediate, dtype=dtype, device=device)
    dY = 0.05 * torch.randn(tokens, hidden, dtype=dtype, device=device)

    # Kernel chain (packed expert-major rows; x rows ARE the packed routing).
    postact, preact = moe_up_proj_act(
        x, w13, offsets, activation="swiglu", gate_up_bias=b13, tune_mode=TuneMode.STATIC
    )
    d_gate_up, _, _ = moe_down_proj_dact(
        dY, w2, preact, offsets, activation="swiglu", tune_mode=TuneMode.STATIC
    )
    d_gate_up = _interleaved_to_concat(d_gate_up, intermediate)

    dw2 = torch.empty(experts, hidden, intermediate, dtype=dtype, device=device)
    plain_gemm(dY.mT, postact, dw2, cu_seqlens_k=offsets, tune_mode=TuneMode.TUNED)
    dw13 = torch.empty(experts, 2 * intermediate, hidden, dtype=dtype, device=device)
    plain_gemm(d_gate_up.mT, x, dw13, cu_seqlens_k=offsets, tune_mode=TuneMode.TUNED)
    dx = torch.empty_like(x)
    for e in range(experts):
        rows = slice(int(offsets[e]), int(offsets[e + 1]))
        dx[rows] = d_gate_up[rows] @ w13[e]
    db13 = torch.stack(
        [d_gate_up[int(offsets[e]) : int(offsets[e + 1])].sum(0) for e in range(experts)]
    )

    # fp32 autograd reference over the same routing.
    xf = x.float().requires_grad_(True)
    w13f = w13.float().requires_grad_(True)
    b13f = b13.float().requires_grad_(True)
    w2f = w2.float().requires_grad_(True)
    zs = []
    outs = []
    for e in range(experts):
        rows = slice(int(offsets[e]), int(offsets[e + 1]))
        z = xf[rows] @ w13f[e].T + b13f[e]
        z.retain_grad()
        zs.append(z)
        a = torch.nn.functional.silu(z[:, :intermediate]) * z[:, intermediate:]
        outs.append(a @ w2f[e].T)
    torch.cat(outs, 0).backward(dY.float())
    d_preact = torch.cat([z.grad for z in zs], 0)

    _assert_close_scaled(d_gate_up, d_preact, "d_gate_up (dact)", dtype)
    _assert_close_scaled(dx, xf.grad, "dx", dtype)
    _assert_close_scaled(dw2, w2f.grad, "dw2", dtype)
    _assert_close_scaled(dw13, w13f.grad, "dw13", dtype)
    _assert_close_scaled(db13, b13f.grad, "db13", dtype)
