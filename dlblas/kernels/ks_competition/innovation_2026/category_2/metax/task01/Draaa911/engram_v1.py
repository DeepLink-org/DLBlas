import functools
import importlib.util
import os
import sysconfig

import torch
import torch.nn as nn


@functools.lru_cache(maxsize=1)
def _load_lib():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python EXT_SUFFIX is unavailable")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "engram_bridge_native" + suffix
    )
    spec = importlib.util.spec_from_file_location("engram_bridge_native", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native forward bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._lib = _load_lib()
        self._fixed_out = None

    def forward(self, ngram_token_ids, multipliers, vocab_sizes, offsets):
        out = torch.empty(
            (2, 4096, 16),
            dtype=torch.int32,
            device=ngram_token_ids.device,
        )
        self._fixed_out = out
        runner = self._lib.make_runner(out)
        self.forward = runner
        return runner(ngram_token_ids, multipliers, vocab_sizes, offsets)


def get_inputs():
    num_tokens, ngram, layers, tables = 4096, 3, 2, 8
    tokens = torch.randint(0, 100000, (num_tokens, ngram), dtype=torch.int32)
    mults = torch.randint(0, 100000, (layers, ngram), dtype=torch.int64)
    vsizes = torch.randint(
        100000, 1000000, (layers, ngram - 1, tables), dtype=torch.int32
    )
    flat = vsizes.view(layers, -1)
    offsets = torch.cat(
        [
            torch.zeros(layers, 1, dtype=torch.int32),
            flat[:, :-1].cumsum(1, dtype=torch.int32),
        ],
        dim=1,
    )
    return [tokens, mults, vsizes, offsets]


def get_init_inputs():
    return []
