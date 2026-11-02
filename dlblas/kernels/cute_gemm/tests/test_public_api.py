"""CPU-only lock of cute_gemm's declared public API surface.

No GPU, no JIT. Three tiers, per the flashinfer house style:

* Tier 1 — the top-level ``__all__`` is pinned exactly, name by name, with
  identity re-exports (``cute_gemm.x is owner_module.x``). Any accidental
  addition, removal, or rename of a public name fails here first.
* Tier 2 — the compute-critical, positionally-called families (GEMM, linear,
  MoE, cross-entropy) get full parameter-name-sequence locks: names, order,
  and no ``*args``/``**kwargs`` (so the lock stays probeable). MoE additionally
  locks the positional/keyword-only boundary and every default value.
* Tier 3 — the remaining names (norm, softmax, classes, transforms) are pinned
  for existence only: the Tier-1 ``__all__`` lock already catches removal.
* Schema lock — every registered ``torch.ops.dlkernel.*`` op whose schema
  carries ``tune_mode`` defaults it to TUNED (0), same as the Python signs.

Changing a signature on purpose means updating one table row here — the
flashinfer convention (contract tables are two-line diffs per added param).
"""

import importlib
import inspect
import re

import pytest
import torch

import cute_gemm
from cute_gemm import TuneMode
from cute_gemm.fused_grouped_gemm import interface as _fgg_interface
from cute_gemm.fused_grouped_gemm.gemm_sm90 import GemmMoeSm90

_ALL_NAMES = {
    "MLP",
    "RMSNorm",
    "RoundingMode",
    "TuneMode",
    "LinearCrossEntropy",
    "chunked_linear_cross_entropy",
    "cross_entropy",
    "cross_entropy_bwd",
    "cross_entropy_fwd",
    "gemm",
    "gemm_add",
    "gemm_symmetric",
    "get_num_sms",
    "gemm_with_activation",
    "gemm_with_activation_grad",
    "hadamard_transform",
    "linear_act_func",
    "linear_func",
    "linear_gated_func",
    "moe_down_proj_dact",
    "moe_up_proj_act",
    "rmsnorm",
    "set_num_sms",
    "softmax",
}


# owner module per name, for the identity half of the Tier-1 lock.
# importlib only: `cute_gemm.softmax` etc. are SHADOWED by the re-exported
# functions of the same name, so attribute access cannot reach the module.
def _mod(name):
    return importlib.import_module(f"cute_gemm.{name}")


_OWNERS = {
    "MLP": _mod("mlp"),
    "RMSNorm": _mod("rmsnorm"),
    "RoundingMode": _mod("rounding"),
    "get_num_sms": _mod("cute_dsl_utils"),
    "set_num_sms": _mod("cute_dsl_utils"),
    "TuneMode": _mod("gemm_config"),
    "chunked_linear_cross_entropy": _mod("linear_cross_entropy"),
    "LinearCrossEntropy": _mod("linear_cross_entropy"),
    "cross_entropy": _mod("cross_entropy"),
    "cross_entropy_bwd": _mod("cross_entropy"),
    "cross_entropy_fwd": _mod("cross_entropy"),
    "gemm": _mod("gemm_interface"),
    "gemm_add": _mod("gemm_interface"),
    "gemm_symmetric": _mod("gemm_symmetric"),
    "gemm_with_activation": _mod("gemm_base"),
    "gemm_with_activation_grad": _mod("gemm_base"),
    "hadamard_transform": _mod("transform"),
    "linear_act_func": _mod("linear"),
    "linear_func": _mod("linear"),
    "linear_gated_func": _mod("linear"),
    "moe_down_proj_dact": _mod("fused_grouped_gemm.interface"),
    "moe_up_proj_act": _mod("fused_grouped_gemm.interface"),
    "rmsnorm": _mod("rmsnorm"),
    "softmax": _mod("softmax"),
}

# Tier 2: parameter-name sequences, order included. A deliberate signature
# change edits exactly one row.
_TIER2_PARAMS = {
    "gemm": (
        "A",
        "B",
        "out",
        "bias",
        "alpha",
        "out_dtype",
        "out_quant_dim",
        "out_transposed",
        "out_per_tensor_scale",
        "cu_seqlens_m",
        "cu_seqlens_k",
        "A_idx",
        "batch_idx_permute",
        "dynamic_scheduler",
        "tune_mode",
        "rounding_mode",
        "sr_seed",
        "concat_layout",
        "split_k",
        "split_k_mode",
    ),
    "gemm_add": (
        "A",
        "B",
        "C",
        "out",
        "bias",
        "alpha",
        "beta",
        "out_dtype",
        "cu_seqlens_m",
        "cu_seqlens_k",
        "A_idx",
        "batch_idx_permute",
        "dynamic_scheduler",
        "tune_mode",
        "concat_layout",
        "split_k",
        "split_k_mode",
    ),
    "gemm_with_activation": (
        "A",
        "B",
        "C",
        "bias",
        "activation",
        "alpha",
        "preact_out",
        "postact_out",
        "out_dtype",
        "postact_dtype",
        "cu_seqlens_m",
        "A_idx",
        "store_preact",
        "dynamic_scheduler",
        "tune_mode",
        "config",
        "concat_layout",
        "split_k",
        "split_k_mode",
    ),
    "gemm_with_activation_grad": (
        "A",
        "B",
        "PreAct",
        "activation",
        "dx_out",
        "postact_out",
        "out_dtype",
        "postact_dtype",
        "colvec_scale",
        "colvec_reduce",
        "cu_seqlens_m",
        "A_idx",
        "dynamic_scheduler",
        "tune_mode",
        "config",
        "split_k",
        "split_k_mode",
    ),
    "gemm_symmetric": (
        "A",
        "B",
        "D",
        "C",
        "tile_count_semaphore",
        "tile_M",
        "tile_N",
        "cluster_M",
        "cluster_N",
        "tile_K",
        "pingpong",
        "persistent",
        "is_dynamic_persistent",
        "max_swizzle_size",
        "alpha",
        "beta",
        "SFA",
        "SFB",
        "bs_format_a",
        "bs_format_b",
    ),
    "linear_func": ("x", "weight", "bias", "fuse_grad_accum", "tune_mode"),
    "linear_act_func": (
        "x",
        "weight",
        "activation",
        "bias",
        "store_preact",
        "fuse_grad_accum",
        "tune_mode",
    ),
    "linear_gated_func": (
        "x",
        "weight",
        "activation",
        "bias",
        "store_preact",
        "fuse_grad_accum",
        "tune_mode",
        "concat_layout",
    ),
    "cross_entropy": (
        "x",
        "target",
        "lse_partial",
        "weight",
        "ignore_index",
        "reduction",
        "inplace_backward",
    ),
    "cross_entropy_fwd": (
        "x",
        "target",
        "target_logit",
        "weight",
        "ignore_index",
        "return_lse",
        "return_dx",
        "inplace_backward",
    ),
    "cross_entropy_bwd": (
        "x",
        "target",
        "dloss",
        "lse",
        "weight",
        "ignore_index",
        "inplace_backward",
    ),
    "chunked_linear_cross_entropy": (
        "x",
        "weight",
        "target",
        "chunk_size",
        "ignore_index",
        "reduction",
        "tune_mode",
        "use_scaled_exp",
    ),
}

_UP_PARAMS = (
    "x",
    "gate_up_weight",
    "expert_offsets",
    "activation",
    "gate_up_layout",
    "token_indices",
    "gate_up_bias",
    "store_preact",
    "tune_mode",
)
_UP_DEFAULTS = {
    "activation": "swiglu",
    "gate_up_layout": "concat",
    "token_indices": None,
    "gate_up_bias": None,
    "store_preact": True,
    "tune_mode": TuneMode.TUNED,
}
_DOWN_PARAMS = (
    "grad_output",
    "down_weight",
    "gate_up_preact",
    "expert_offsets",
    "activation",
    "token_indices",
    "router_weights",
    "compute_router_weight_grad",
    "tune_mode",
)
_DOWN_DEFAULTS = {
    "activation": "swiglu",
    "token_indices": None,
    "router_weights": None,
    "compute_router_weight_grad": False,
    "tune_mode": TuneMode.TUNED,
}
_ACTIVATIONS = {
    "swiglu",
    "geglu",
    "reglu",
    "silu",
    "relu",
    "relu_sq",
    "gelu_tanh_approx",
}


def test_top_level_all_is_exactly_the_declared_surface():
    """Tier 1: the declared public surface, no more, no less."""
    assert set(cute_gemm.__all__) == _ALL_NAMES


@pytest.mark.parametrize("name", sorted(_ALL_NAMES))
def test_public_name_is_reexported_from_its_owner(name):
    """Tier 1: every public name is the owner module's own object."""
    assert hasattr(cute_gemm, name), f"{name} missing from cute_gemm"
    assert getattr(cute_gemm, name) is getattr(_OWNERS[name], name)


@pytest.mark.parametrize("name", sorted(_TIER2_PARAMS))
def test_signature_contract(name):
    """Tier 2: parameter names and order are pinned for the compute families."""
    fn = getattr(cute_gemm, name)
    params = inspect.signature(fn).parameters
    assert tuple(params) == _TIER2_PARAMS[name], (
        f"{name} signature drifted; update the table if intentional"
    )
    assert not any(
        v.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        for v in params.values()
    ), f"{name} must stay free of *args/**kwargs (the lock must stay probeable)"


# The registered torch.ops.dlkernel ops whose schema carries ``tune_mode`` —
# exactly the GEMM family of gemm_interface.py. A deliberate re-flip of the
# default edits this one row.
_TUNE_MODE_OPS = frozenset(
    {
        "gemm",
        "gemm_out",
        "gemm_add",
        "gemm_add_out",
        "gemm_add_inplace",
        "gemm_act_out",
        "gemm_gated_out",
        "gemm_dact_out",
        "gemm_dgated_out",
        "gemm_rms_out",
        "gemm_norm_act_out",
        "gemm_norm_gated_out",
        "gemm_quant_out",
    }
)


def test_torch_op_schemas_default_tune_mode_to_tuned():
    """Every dlkernel op with ``tune_mode`` in its schema defaults it to
    TUNED (0), like the Python signatures. A schema left at RUNNING (2) sends
    direct ``torch.ops.dlkernel.*`` calls that omit the mode silently into
    RUNNING — the schema string, not the Python default, is the wire."""
    registered = {
        schema.name.split("::", 1)[1]
        for schema in torch._C._jit_get_all_schemas()
        if schema.name.startswith("dlkernel::")
        and any(arg.name == "tune_mode" for arg in schema.arguments)
    }
    assert registered == _TUNE_MODE_OPS, (
        "the tune_mode op set drifted; update _TUNE_MODE_OPS if intentional"
    )
    want = str(int(TuneMode.TUNED))
    for name in sorted(registered):
        schema = str(getattr(torch.ops.dlkernel, name).default._schema)
        found = re.findall(r"\btune_mode=(-?\d+)", schema)
        assert found == [want], (
            f"dlkernel::{name}: schema default tune_mode={found} != {want}; "
            "schema and Python signature diverged"
        )


@pytest.mark.parametrize("fname", ["gemm", "gemm_add"])
def test_python_signature_default_tune_mode_is_tuned(fname):
    """The Python half of the same contract: the public entries still default
    to TUNED, so a re-flip cannot slip past this lock from either side."""
    default = inspect.signature(getattr(cute_gemm, fname)).parameters["tune_mode"].default
    assert default == TuneMode.TUNED, (
        f"{fname}: Python default tune_mode={default!r} re-flipped; update this lock"
    )


@pytest.mark.parametrize(
    "fn,params,defaults,n_positional",
    [
        pytest.param(_fgg_interface.moe_up_proj_act, _UP_PARAMS, _UP_DEFAULTS, 3, id="up"),
        pytest.param(_fgg_interface.moe_down_proj_dact, _DOWN_PARAMS, _DOWN_DEFAULTS, 4, id="down"),
    ],
)
def test_moe_signature_defaults_and_boundary(fn, params, defaults, n_positional):
    """MoE gets the fullest lock: boundary between positional and keyword-only,
    every default value, on top of the Tier-2 name sequence."""
    p = inspect.signature(fn).parameters
    assert tuple(p) == params
    for name in params[:n_positional]:
        assert p[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert p[name].default is inspect.Parameter.empty, f"{name} must stay required"
    for name in params[n_positional:]:
        assert p[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{fn.__name__}.{name} must stay keyword-only"
        )
    assert {
        k: v.default for k, v in p.items() if v.default is not inspect.Parameter.empty
    } == defaults


@pytest.mark.parametrize(
    "entry,method_name,n_positional",
    [
        pytest.param(_fgg_interface.moe_up_proj_act, "up_proj_act", 3, id="up"),
        pytest.param(_fgg_interface.moe_down_proj_dact, "down_proj_dact", 4, id="down"),
    ],
)
def test_moe_entry_forwards_every_argument_and_adds_nothing(
    monkeypatch, entry, method_name, n_positional
):
    """The public MoE entries forward every argument, rewrite none, add none."""
    sig = inspect.signature(entry)
    names = tuple(sig.parameters)
    values = [object() for _ in names]
    calls = []
    monkeypatch.setattr(
        GemmMoeSm90,
        method_name,
        staticmethod(lambda *a, **kw: calls.append((a, kw)) or "result"),
    )
    assert entry(*values[:n_positional]) == "result"
    args, kwargs = calls[0]
    assert args == tuple(values[:n_positional])
    for name in names[n_positional:]:
        assert kwargs[name] == sig.parameters[name].default, f"{name} forwarded as {kwargs[name]!r}"
    kw_names = names[n_positional:]
    entry(*values[:n_positional], **dict(zip(kw_names, values[n_positional:])))
    args, kwargs = calls[1]
    assert args == tuple(values[:n_positional])
    for name, value in zip(kw_names, values[n_positional:]):
        assert kwargs[name] is value, f"{name} was rewritten"
    assert set(kwargs) == set(kw_names), "the entry added extra arguments"


def test_public_surface_has_no_internal_tuning_knobs():
    """Kernel-level knobs must not appear on the public primitives."""
    forbidden = {
        "tile_m",
        "tile_n",
        "cluster_m",
        "num_stages",
        "swizzle",
        "kernel_variant",
        "variant",
        "schedule",
        "autotune",
        "backend",
        "sm_base",
    }
    for name in ("moe_up_proj_act", "moe_down_proj_dact"):
        fn = getattr(cute_gemm, name)
        assert not forbidden & set(inspect.signature(fn).parameters)


def test_activation_tokens_are_the_declared_whitelist():
    assert set(GemmMoeSm90._ACTIVATIONS) == _ACTIVATIONS


def test_unknown_activation_is_rejected_by_name():
    x = torch.zeros(2, 16, dtype=torch.bfloat16)
    w = torch.zeros(1, 16, 16, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="activation"):
        cute_gemm.moe_up_proj_act(x, w, offsets, activation="bogus")
