# Copyright (c) 2025, DeepLink.

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import triton


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "dlblas" / "kernels" / "mattersim_opt.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "mattersim_opt_bench_module", _MODULE_PATH
)
_KERNELS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KERNELS)

linear_bias_activation = _KERNELS.linear_bias_activation
scatter_add_2d_dim0 = _KERNELS.scatter_add_2d_dim0
two_path_two_layer_mlp = _KERNELS.two_path_two_layer_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _bench_pair(name, triton_fn, torch_fn, warmup=50, rep=200):
    opt_ms = triton.testing.do_bench(triton_fn, warmup=warmup, rep=rep)
    torch_ms = triton.testing.do_bench(torch_fn, warmup=warmup, rep=rep)
    speedup = torch_ms / opt_ms if opt_ms > 0 else float("inf")
    print(
        f"{name:28s} opt={opt_ms:.4f} ms torch={torch_ms:.4f} ms speedup={speedup:.3f}x"
    )


def bench_linear(dtype=torch.float32):
    torch.manual_seed(0)
    x = torch.randn(8192, 384, device=DEVICE, dtype=dtype)
    weight = torch.randn(128, 384, device=DEVICE, dtype=dtype)
    bias = torch.randn(128, device=DEVICE, dtype=dtype)
    _bench_pair(
        "mattersim_opt_linear_bias_swish",
        lambda: linear_bias_activation(x, weight, bias, activation="swish"),
        lambda: F.silu(F.linear(x, weight, bias)),
    )


def bench_two_path(dtype=torch.float32):
    torch.manual_seed(1)
    x = torch.randn(8192, 384, device=DEVICE, dtype=dtype)
    w1g = torch.randn(128, 384, device=DEVICE, dtype=dtype)
    b1g = torch.randn(128, device=DEVICE, dtype=dtype)
    w2g = torch.randn(128, 128, device=DEVICE, dtype=dtype)
    b2g = torch.randn(128, device=DEVICE, dtype=dtype)
    w1s = torch.randn(128, 384, device=DEVICE, dtype=dtype)
    b1s = torch.randn(128, device=DEVICE, dtype=dtype)
    w2s = torch.randn(128, 128, device=DEVICE, dtype=dtype)
    b2s = torch.randn(128, device=DEVICE, dtype=dtype)

    def torch_ref():
        g = F.silu(F.linear(x, w1g, b1g))
        g = F.silu(F.linear(g, w2g, b2g))
        s = F.silu(F.linear(x, w1s, b1s))
        s = torch.sigmoid(F.linear(s, w2s, b2s))
        return g * s

    _bench_pair(
        "mattersim_opt_two_path_mlp",
        lambda: two_path_two_layer_mlp(x, w1g, b1g, w2g, b2g, w1s, b1s, w2s, b2s),
        torch_ref,
    )


def bench_scatter(dtype=torch.float32):
    torch.manual_seed(2)
    src = torch.randn(50000, 128, device=DEVICE, dtype=dtype)
    index = torch.randint(0, 12000, (src.shape[0],), device=DEVICE, dtype=torch.long)

    def torch_ref():
        out = torch.zeros(12000, src.shape[1], dtype=src.dtype, device=src.device)
        return out.scatter_add_(0, index.view(-1, 1).expand_as(src), src)

    _bench_pair(
        "mattersim_opt_scatter_add",
        lambda: scatter_add_2d_dim0(src, index, dim_size=12000),
        torch_ref,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=["all", "linear", "two-path", "scatter"],
        default="all",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    args = parser.parse_args()

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if args.op in ("all", "linear"):
        bench_linear(dtype)
    if args.op in ("all", "two-path"):
        bench_two_path(dtype)
    if args.op in ("all", "scatter"):
        bench_scatter(dtype)


if __name__ == "__main__":
    main()
