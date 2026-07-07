# Indexer Benchmark — Performance measurement for the 4-kernel AscendC implementation
# Measures end-to-end latency for prefill and decode scenarios

import sys
import os
import time
import torch
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ascendc.indexer_launcher import IndexerLauncher
from test.torch_ref.indexer_torch import get_init_inputs


def run_benchmark(launcher, x, qr, start_pos, offset, warmup=10, repeat=100):
    """Run performance benchmark and return latency statistics in milliseconds."""
    # Warmup
    for _ in range(warmup):
        launcher.forward(x, qr, start_pos, offset)
    torch.npu.synchronize()

    # Benchmark
    latencies = []
    for _ in range(repeat):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        launcher.forward(x, qr, start_pos, offset)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = sorted(latencies)
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "median_ms": latencies[len(latencies) // 2],
        "p90_ms": latencies[int(len(latencies) * 0.9)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "repeat": repeat,
    }


def main():
    parser = argparse.ArgumentParser(description="Indexer performance benchmark")
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    torch.manual_seed(42)
    init_args, wq_b_w, w_proj_w, kv_cache_base, freqs_cis_base = get_init_inputs(args.device)

    def build_launcher(max_seq_len: int):
        """Build a launcher with appropriate max_seq_len for the test case."""
        # Rebuild freqs_cis for the required max_seq_len
        rope_theta = 10000.0
        rd = init_args["rope_head_dim"]
        freqs = 1.0 / (rope_theta ** (torch.arange(0, rd, 2, device=args.device)[:rd // 2].float() / rd))
        t = torch.arange(max_seq_len, device=args.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Rebuild kv_cache for the required max_seq_len
        compress_ratio = init_args["compress_ratio"]
        max_kv_len = max_seq_len // compress_ratio
        kv_cache = torch.randn(init_args["max_batch_size"], max_kv_len,
                               init_args["head_dim"], dtype=torch.bfloat16, device=args.device)

        launcher = IndexerLauncher(
            dim=init_args["dim"],
            n_heads=init_args["n_heads"],
            head_dim=init_args["head_dim"],
            rope_head_dim=init_args["rope_head_dim"],
            index_topk=init_args["index_topk"],
            q_lora_rank=init_args["q_lora_rank"],
            compress_ratio=init_args["compress_ratio"],
            max_seq_len=max_seq_len,
            max_batch_size=init_args["max_batch_size"],
            device=args.device,
        )
        launcher.load_weights(
            wq_weight=wq_b_w,
            w_proj_weight=w_proj_w,
            kv_cache=kv_cache,
            freqs_cis=freqs_cis,
        )
        return launcher

    results = {}

    # --- Prefill Benchmarks ---
    print("Indexer Performance Benchmark")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Avg(ms)':>10} {'Min(ms)':>10} {'Median(ms)':>12} {'P90(ms)':>10}")
    print("-" * 70)

    prefill_shapes = [(2, 64, 1024), (2, 512, 1024), (2, 4096, 4096)]
    for B, S, max_sl in prefill_shapes:
        name = f"Prefill B={B} S={S}"
        x = torch.randn(B, S, init_args["dim"], dtype=torch.bfloat16, device=args.device)
        qr = torch.randn(B, S, init_args["q_lora_rank"], dtype=torch.bfloat16, device=args.device)

        try:
            launcher = build_launcher(max_sl)
            stats = run_benchmark(launcher, x, qr, 0, 0, args.warmup, args.repeat)
            results[name] = stats
            print(f"{name:<25} {stats['avg_ms']:>10.4f} {stats['min_ms']:>10.4f} {stats['median_ms']:>12.4f} {stats['p90_ms']:>10.4f}")
        except Exception as e:
            print(f"{name:<25} {'ERROR':>10} - {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # --- Decode Benchmarks ---
    decode_cases = [(2, 64, 1024), (2, 512, 1024), (2, 4096, 5000)]
    for B, start_pos, max_sl in decode_cases:
        name = f"Decode B={B} pos={start_pos}"
        x = torch.randn(B, 1, init_args["dim"], dtype=torch.bfloat16, device=args.device)
        qr = torch.randn(B, 1, init_args["q_lora_rank"], dtype=torch.bfloat16, device=args.device)

        try:
            launcher = build_launcher(max_sl)
            stats = run_benchmark(launcher, x, qr, start_pos, 0, args.warmup, args.repeat)
            results[name] = stats
            print(f"{name:<25} {stats['avg_ms']:>10.4f} {stats['min_ms']:>10.4f} {stats['median_ms']:>12.4f} {stats['p90_ms']:>10.4f}")
        except Exception as e:
            print(f"{name:<25} {'ERROR':>10} - {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    print("-" * 70)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Summary
    print("\nSummary:")
    for name, stats in results.items():
        if "error" not in stats:
            print(f"  {name}: {stats['avg_ms']:.4f} ms (avg)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
