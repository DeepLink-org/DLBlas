#!/usr/bin/env python3
"""
Benchmark: Triton kernels (ModelNew) vs PyTorch reference (Model) — level3 算子性能对比

执行方式见 README.md（同目录）。
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
TRITON_DIR = os.path.normpath(os.path.join(_HERE, "../dlblas/kernels/kernelswift_triton/level3"))
TORCH_DIR  = os.path.normpath(os.path.join(_HERE, "../dlblas/kernels/kernelswift_torch/level3"))

# 候选算子：friendly name → 文件名（两目录文件名相同）
CANDIDATES = {
    "sparse_attn":      "9_sparse_attn.py",
    "hc_split_sinkhorn":"17_hc_split_sinkhorn.py",
    "hc_post":          "12_post.py",
    "indexer":          "11_indexer.py",
    "MTPBlock":         "10_MTPBlock.py",
    "act_quant_kernel": "18_act_quant_fp8.py",
}


def load_module(mod_name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    mod.__name__ = mod_name   # 避免触发 __main__ 块
    spec.loader.exec_module(mod)
    return mod


def do_bench(fn, warmup: int = 25, rep: int = 100) -> float:
    """返回中位延迟（ms）。优先用 triton.testing.do_bench，否则手动计时。"""
    try:
        import triton.testing
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    except Exception:
        pass

    # 校准：估算单次调用耗时，计算内层重复次数使每次测量 ≥ 1 ms，降低计时误差
    _t0, _t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    _t0.record()
    for _ in range(5):
        fn()
    _t1.record()
    torch.cuda.synchronize()
    _est_ms = _t1.elapsed_time(_t0) / 5          # 单次估算（ms）
    # n_inner = max(1, int(1.0 / max(_est_ms, 1e-6)))  # 内层重复次数，目标 ~1 ms/次
    n_inner = 200

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_ev   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for i in range(rep):
        start_ev[i].record()
        for _ in range(n_inner):
            fn()
        end_ev[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) / n_inner for s, e in zip(start_ev, end_ev))
    return times[len(times) // 2]


def to_cuda(inputs: list) -> list:
    return [t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t
            for t in inputs]


def bench_one(mod, class_name: str, warmup: int, rep: int) -> float:
    if not hasattr(mod, class_name):
        raise AttributeError(f"Module has no class '{class_name}'")

    init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = getattr(mod, class_name)(*init_inputs)
    model = model.cuda().eval()

    fwd_inputs = to_cuda(mod.get_inputs())
    fn = lambda: model(*fwd_inputs)

    with torch.no_grad():
        fn()  # 触发 JIT / autotune，不计入计时
    torch.cuda.synchronize()

    with torch.no_grad():
        return do_bench(fn, warmup=warmup, rep=rep)


def run_benchmark(kernel_name: str, warmup: int, rep: int) -> dict:
    fname = CANDIDATES[kernel_name]
    result = {
        "name":       kernel_name,
        "file":       fname,
        "triton_ms":  None,
        "torch_ms":   None,
        "speedup":    None,
        "triton_err": None,
        "torch_err":  None,
    }

    try:
        mod = load_module(f"{kernel_name}_triton", os.path.join(TRITON_DIR, fname))
        result["triton_ms"] = bench_one(mod, "ModelNew", warmup, rep)
    except Exception as e:
        result["triton_err"] = f"{type(e).__name__}: {e}"
        if os.environ.get("BENCH_VERBOSE"):
            traceback.print_exc()

    try:
        mod = load_module(f"{kernel_name}_torch", os.path.join(TORCH_DIR, fname))
        result["torch_ms"] = bench_one(mod, "Model", warmup, rep)
    except Exception as e:
        result["torch_err"] = f"{type(e).__name__}: {e}"
        if os.environ.get("BENCH_VERBOSE"):
            traceback.print_exc()

    if result["triton_ms"] and result["torch_ms"]:
        result["speedup"] = result["torch_ms"] / result["triton_ms"]

    return result


def print_table(results: list) -> None:
    name_w = max(len(r["name"]) for r in results) + 2
    w_ms   = 12
    w_sp   = 10

    header = (
        f"{'Kernel':<{name_w}}"
        f"{'PyTorch(ms)':>{w_ms}}"
        f"{'Triton(ms)':>{w_ms}}"
        f"{'Speedup':>{w_sp}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        torch_str  = f"{r['torch_ms']:.4f}"  if r["torch_ms"]  else "ERROR"
        triton_str = f"{r['triton_ms']:.4f}" if r["triton_ms"] else "ERROR"
        sp_str     = f"{r['speedup']:.2f}x"  if r["speedup"] is not None else "N/A"

        print(
            f"{r['name']:<{name_w}}"
            f"{torch_str:>{w_ms}}"
            f"{triton_str:>{w_ms}}"
            f"{sp_str:>{w_sp}}"
        )
        if r["triton_err"]:
            print(f"  [triton error]  {r['triton_err']}")
        if r["torch_err"]:
            print(f"  [torch  error]  {r['torch_err']}")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton vs PyTorch level3 kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "候选算子:",
            *[f"  {k:<22} {v}" for k, v in CANDIDATES.items()],
        ]),
    )
    parser.add_argument(
        "kernels", nargs="*", metavar="KERNEL",
        help="要测试的算子（默认全部）",
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有候选算子后退出")
    parser.add_argument("--warmup", type=int, default=25,
                        help="预热迭代次数（默认 25）")
    parser.add_argument("--rep",    type=int, default=100,
                        help="计时重复次数（默认 100）")
    args = parser.parse_args()

    if args.list:
        print("候选算子:")
        for name, fname in CANDIDATES.items():
            print(f"  {name:<22} -> {fname}")
        return

    selected = args.kernels or list(CANDIDATES.keys())
    unknown  = [k for k in selected if k not in CANDIDATES]
    if unknown:
        print(f"未知算子: {', '.join(unknown)}")
        print(f"可用算子: {', '.join(CANDIDATES)}")
        sys.exit(1)

    print(f"\nBenchmarking {len(selected)} kernel(s)  "
          f"[warmup={args.warmup}, rep={args.rep}]\n")

    results = []
    for name in selected:
        print(f"  {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        r  = run_benchmark(name, warmup=args.warmup, rep=args.rep)
        elapsed = time.perf_counter() - t0
        if r["speedup"] is not None:
            summary = f"speedup {r['speedup']:.2f}x  ({elapsed:.1f}s)"
        else:
            summary = f"partial/error  ({elapsed:.1f}s)"
        print(f"\r  {name:<22} {summary}")
        results.append(r)

    print()
    print_table(results)


if __name__ == "__main__":
    main()
