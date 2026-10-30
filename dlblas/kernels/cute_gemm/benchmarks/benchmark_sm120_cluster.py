"""Benchmark reduction kernels with cluster support on SM120.

Usage:
    python benchmarks/benchmark_sm120_cluster.py
    python benchmarks/benchmark_sm120_cluster.py --dtype Float32
    python benchmarks/benchmark_sm120_cluster.py --M 8192 --N 65536
"""

import argparse
import os
import time

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")


import torch
import torch.nn.functional as F
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from cute_gemm.rmsnorm import rmsnorm_fwd, rmsnorm_bwd, rmsnorm
from cute_gemm.softmax import softmax
from cute_gemm.cross_entropy import cross_entropy_fwd, cross_entropy


def io_bytes(kernel, M, N, db):
    """Compute I/O bytes for bandwidth calculation.

    Args:
        kernel: kernel name
        M: batch dimension
        N: hidden dimension
        db: dtype bytes (2 for bf16/fp16, 4 for fp32)
    """
    if kernel == "rmsnorm_fwd":
        # read x + read w + write out
        return 2 * M * N * db + N * 4
    elif kernel == "rmsnorm_fwd_res":
        # read x + read res + write out + write res_out
        return 4 * M * N * db + N * 4
    elif kernel == "rmsnorm_bwd":
        # read x + read dy + read w + write dx
        return 3 * M * N * db + N * 4
    elif kernel == "softmax_fwd":
        # read x + write out
        return 2 * M * N * db
    elif kernel == "softmax_bwd":
        # read dy + read y + write dx
        return 3 * M * N * db
    elif kernel == "ce_fwd":
        # read x + read target + write loss
        return M * N * db + M * 12
    elif kernel == "ce_bwd":
        # read x + read target + read dloss + read lse + write dx
        return 2 * M * N * db + M * 16
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def bench_one(fn, io, warmup=5, rep=50):
    time.sleep(0.1)
    t = do_bench(fn, warmup=warmup, rep=rep)
    bw = round(io / (t / 1000) / 1e9)
    return t, bw


def _compile_refs():
    """Compile torch reference implementations (call once to warm up)."""
    refs = {
        "rmsnorm": torch.compile(
            lambda x, w: x / (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt() * w
        ),
        "softmax": torch.compile(lambda x: F.softmax(x, dim=-1)),
        "ce": torch.compile(lambda x, t: F.cross_entropy(x, t, reduction="none")),
    }
    return refs


def run_sweep(M, N_vals, dtype, warmup=5, rep=50):
    db = dtype.itemsize
    results = []

    # Warm up compiled references
    refs = _compile_refs()
    x_ = torch.randn(M, 4096, device="cuda", dtype=dtype)
    w_ = torch.randn(4096, device="cuda", dtype=torch.float32)
    t_ = torch.randint(0, 4096, (M,), device="cuda")
    for _ in range(3):
        refs["rmsnorm"](x_, w_)
        refs["softmax"](x_)
        refs["ce"](x_, t_)
    del x_, w_, t_

    for N in N_vals:
        torch.cuda.empty_cache()
        row = {"N": N}

        x = torch.randn(M, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=torch.float32)
        res = torch.randn(M, N, device="cuda", dtype=dtype)

        # rmsnorm fwd
        try:
            rmsnorm_fwd(x, w, eps=1e-6)
            fn = lambda x=x, w=w: rmsnorm_fwd(x, w, eps=1e-6)
            t, bw = bench_one(fn, io_bytes("rmsnorm_fwd", M, N, db))
            row["rmsnorm_fwd"] = bw
        except Exception:
            row["rmsnorm_fwd"] = "SMEM"

        # rmsnorm fwd (torch.compile ref)
        fn_ref = lambda x=x, w=w: refs["rmsnorm"](x, w)
        _, bw_ref = bench_one(fn_ref, io_bytes("rmsnorm_fwd", M, N, db))
        row["rmsnorm_fwd_ref"] = bw_ref

        # rmsnorm fwd+res
        try:
            rmsnorm_fwd(x, w, residual=res, eps=1e-6)
            fn = lambda x=x, w=w, res=res: rmsnorm_fwd(x, w, residual=res, eps=1e-6)
            t, bw = bench_one(fn, io_bytes("rmsnorm_fwd_res", M, N, db))
            row["rmsnorm_fwd_res"] = bw
        except Exception:
            row["rmsnorm_fwd_res"] = "SMEM"

        # rmsnorm bwd
        try:
            y = rmsnorm(x.requires_grad_(True), w.requires_grad_(True), eps=1e-6)
            dy = torch.randn_like(y)
            rstd = torch.randn(M, device="cuda", dtype=torch.float32)
            rmsnorm_bwd(x, w, dy, rstd)
            fn = lambda x=x, w=w, dy=dy, rstd=rstd: rmsnorm_bwd(x, w, dy, rstd)
            t, bw = bench_one(fn, io_bytes("rmsnorm_bwd", M, N, db))
            row["rmsnorm_bwd"] = bw
        except Exception:
            row["rmsnorm_bwd"] = "SMEM"

        # softmax fwd
        xs = (0.1 * torch.randn(M, N, device="cuda", dtype=dtype)).requires_grad_(True)
        try:
            softmax(xs)
            fn = lambda xs=xs: softmax(xs)
            t, bw = bench_one(fn, io_bytes("softmax_fwd", M, N, db))
            row["softmax_fwd"] = bw
        except Exception:
            row["softmax_fwd"] = "SMEM"

        # softmax fwd (torch.compile ref)
        fn_ref = lambda xs=xs: refs["softmax"](xs)
        _, bw_ref = bench_one(fn_ref, io_bytes("softmax_fwd", M, N, db))
        row["softmax_fwd_ref"] = bw_ref

        # softmax bwd
        try:
            ys = softmax(xs)
            dys = torch.randn_like(ys)
            fn = lambda ys=ys, xs=xs, dys=dys: torch.autograd.grad(
                ys, xs, grad_outputs=dys, retain_graph=True
            )
            t, bw = bench_one(fn, io_bytes("softmax_bwd", M, N, db))
            row["softmax_bwd"] = bw
        except Exception:
            row["softmax_bwd"] = "SMEM"

        # CE fwd
        tgt = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
        xce = x.contiguous()
        try:
            cross_entropy_fwd(xce, tgt)
            fn = lambda xce=xce, tgt=tgt: cross_entropy_fwd(xce, tgt)
            t, bw = bench_one(fn, io_bytes("ce_fwd", M, N, db))
            row["ce_fwd"] = bw
        except Exception:
            row["ce_fwd"] = "SMEM"

        # CE fwd (torch.compile ref)
        fn_ref = lambda xce=xce, tgt=tgt: refs["ce"](xce, tgt)
        _, bw_ref = bench_one(fn_ref, io_bytes("ce_fwd", M, N, db))
        row["ce_fwd_ref"] = bw_ref

        # CE bwd
        xce2 = (0.1 * torch.randn(M, N, device="cuda", dtype=dtype)).requires_grad_(True)
        try:
            loss = cross_entropy(xce2, tgt, reduction="none")
            dloss = torch.randn(M, device="cuda", dtype=torch.float32)
            torch.autograd.grad(loss, xce2, grad_outputs=dloss, retain_graph=True)
            fn = lambda loss=loss, xce2=xce2, dloss=dloss: torch.autograd.grad(
                loss, xce2, grad_outputs=dloss, retain_graph=True
            )
            t, bw = bench_one(fn, io_bytes("ce_bwd", M, N, db))
            row["ce_bwd"] = bw
        except Exception:
            row["ce_bwd"] = "SMEM"

        results.append(row)
        del x, w, res, xs, xce, tgt, xce2
        torch.cuda.empty_cache()

    return results


def print_table(results, dtype_name):
    # cute_gemm-only columns
    q_cols = [
        "rmsnorm_fwd", "rmsnorm_fwd_res", "rmsnorm_bwd",
        "softmax_fwd", "softmax_bwd", "ce_fwd", "ce_bwd",
    ]
    q_headers = [
        "rmsnorm fwd", "rmsnorm fwd+res", "rmsnorm bwd",
        "softmax fwd", "softmax bwd", "CE fwd", "CE bwd",
    ]
    print(f"\n=== {dtype_name} — cute_gemm (GB/s) ===\n")
    print("| N | " + " | ".join(q_headers) + " |")
    print("|---" * (len(q_headers) + 1) + "|")
    for row in results:
        vals = [str(row.get(c, "-")) for c in q_cols]
        print(f"| {row['N']} | " + " | ".join(vals) + " |")

    # Comparison table (cute_gemm vs torch.compile)
    cmp = [
        ("rmsnorm fwd", "rmsnorm_fwd", "rmsnorm_fwd_ref"),
        ("softmax fwd", "softmax_fwd", "softmax_fwd_ref"),
        ("CE fwd", "ce_fwd", "ce_fwd_ref"),
    ]
    print(f"\n=== {dtype_name} — cute_gemm vs torch.compile (GB/s) ===\n")
    header = "| N |"
    sep = "|---|"
    for label, _, _ in cmp:
        header += f" {label} cute_gemm | {label} compile |"
        sep += "---|---|"
    print(header)
    print(sep)
    for row in results:
        line = f"| {row['N']} |"
        for _, qk, rk in cmp:
            qv = row.get(qk, "-")
            rv = row.get(rk, "-")
            line += f" {qv} | {rv} |"
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark reduction kernels with cluster")
    parser.add_argument("--M", default=4096, type=int)
    parser.add_argument(
        "--N", default=None, type=int, help="Single N value (default: sweep)"
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
        default=None,
        help="Single dtype (default: sweep bf16 + fp32)",
    )
    args = parser.parse_args()

    N_vals = [args.N] if args.N else [4096, 8192, 16384, 32768, 65536, 131072]

    if args.dtype:
        dtypes = [(str(args.dtype), cutlass_torch.dtype(args.dtype))]
    else:
        dtypes = [("BFloat16", torch.bfloat16), ("Float32", torch.float32)]

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, SM {p.major}.{p.minor}, {p.total_memory // 2**30} GB")
    print(f"M={args.M}")

    for dtype_name, dtype in dtypes:
        results = run_sweep(args.M, N_vals, dtype)
        print_table(results, dtype_name)
