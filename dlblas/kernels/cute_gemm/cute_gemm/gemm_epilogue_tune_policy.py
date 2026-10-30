"""Epilogue-specific GEMM autotune policy.

The class is intentionally a thin leaf: variable-length identity and selector
behaviour live in :class:`cute_gemm.varlen_selector.VarlenGemmTunePolicy`, while
this module contributes only explicit epilogue tensor roles and host legality.
"""

from __future__ import annotations

from collections.abc import Iterable

from cute_gemm.epilogue.legality import prune_epilogue_configs
from cute_gemm.varlen_selector import VarlenGemmTunePolicy


class GemmEpilogueTunePolicy(VarlenGemmTunePolicy):
    """Add epilogue/mod constraints on top of the varlen GEMM policy."""

    def __init__(
        self,
        mod,
        transform_a=None,
        *,
        ragged_m_out_names: Iterable[str] | None = None,
        opaque_tensor_names: Iterable[str] = (),
        routed_m_tensor_names: Iterable[str] = ("mColVecBroadcast",),
    ) -> None:
        super().__init__()
        self.mod = mod
        self.transform_a = transform_a
        sink_names = tuple(
            name for name, op in mod.sinks.items() if hasattr(op, "sink_alloc_shape")
        )
        declared_outputs = tuple(mod.outputs)
        if ragged_m_out_names is None:
            ragged_m_out_names = ("out", "C", "D", *declared_outputs)
        self._ragged_m_out_names = frozenset(ragged_m_out_names)
        self._opaque_tensor_names = frozenset((*sink_names, *opaque_tensor_names))
        self._routed_m_tensor_names = frozenset(routed_m_tensor_names)

    def bind_signature(self, arg_names, arg_defaults) -> None:
        super().bind_signature(arg_names, arg_defaults)
        self._opaque_tensor_names = frozenset(
            (
                *self._opaque_tensor_names,
                *(name for name in self.arg_names if name.startswith("ta__")),
            )
        )

    def prepare_candidates(self, configs, named_args, kwargs):
        configs = super().prepare_candidates(configs, named_args, kwargs)
        return prune_epilogue_configs(self.mod, self.transform_a, configs, named_args, **kwargs)

    def running_configs(self, args, kwargs, configs):
        from cute_gemm.gemm_running import running_configs

        if self.transform_a is not None:
            return None
        return running_configs(
            self,
            args,
            kwargs,
            configs,
            mod=self.mod,
            b_kn_default=False,
        )


__all__ = ["GemmEpilogueTunePolicy"]
