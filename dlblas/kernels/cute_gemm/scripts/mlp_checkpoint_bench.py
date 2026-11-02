"""Benchmark MLP activation checkpointing: memory usage and GEMM kernel counts.

Compares three modes:
  - normal: default, saves preact for backward
  - checkpoint: torch.utils.checkpoint, replays entire forward (8 GEMMs)
  - recompute: built-in recompute=True, replays only fc1 (7 GEMMs)
  - checkpoint+compile: checkpoint under torch.compile (still 8 GEMMs)
"""
import argparse
import gc
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.profiler import profile, ProfilerActivity
from cute_gemm.mlp import MLP

device = "cuda"
dtype = torch.bfloat16

MODES = ["normal", "checkpoint", "recompute", "checkpoint+compile"]


# --- Memory profiling (stacked layers) ---

def report_mem(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e6
    print(f"  {label:45s}  {alloc:8.1f} MB")


class StackedMLP(nn.Module):
    def __init__(self, dim, hidden, n_layers, activation, use_checkpoint=False, recompute=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLP(dim, hidden, activation=activation, device=device, dtype=dtype,
                 tuned=False, recompute=recompute)
             for _ in range(n_layers)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                x = cp.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


def run_memory(mode, B, S, D, hidden, n_layers, activation):
    use_checkpoint = mode == "checkpoint"
    recompute = mode == "recompute"
    tag = {"normal": "normal", "checkpoint": "torch.utils.checkpoint",
           "recompute": "recompute=True"}[mode]
    print(f"\n{'='*65}")
    print(f"  {tag} ({n_layers} layers)")
    print(f"{'='*65}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    model = StackedMLP(D, hidden, n_layers, activation,
                       use_checkpoint=use_checkpoint, recompute=recompute)
    report_mem("model params")

    x = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
    report_mem("+ input")
    baseline = torch.cuda.memory_allocated()

    out = model(x)
    fwd_mem = torch.cuda.memory_allocated()
    report_mem("+ forward (all layers)")
    activation_mem = (fwd_mem - baseline) / 1e6
    print(f"  {'  -> activation memory':45s}  {activation_mem:8.1f} MB")

    loss = (out ** 2).sum()
    loss.backward()
    report_mem("+ backward")

    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"  {'peak memory':45s}  {peak:8.1f} MB")

    del model, x, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return activation_mem, peak


# --- Kernel profiling (single layer) ---

def run_kernel_profile(mode, B, S, D, hidden, activation):
    use_compile = mode.endswith("+compile")
    base_mode = mode.split("+")[0]
    recompute = base_mode == "recompute"

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model = MLP(D, hidden, activation=activation, device=device, dtype=dtype,
                tuned=False, recompute=recompute)
    if use_compile:
        model = torch.compile(model)
    x = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)

    def run_fwd_bwd():
        if base_mode == "checkpoint":
            out = cp.checkpoint(model, x, use_reentrant=False)
        else:
            out = model(x)
        (out ** 2).sum().backward()

    # Warmup
    run_fwd_bwd()
    x.grad = None
    model.zero_grad()
    if use_compile:
        run_fwd_bwd()
        x.grad = None
        model.zero_grad()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run_fwd_bwd()
        torch.cuda.synchronize()

    return prof.key_averages()


def print_kernel_events(events):
    cuda_events = [
        e for e in events
        if e.device_time_total > 0
        and "memset" not in e.key.lower()
        and "memcpy" not in e.key.lower()
    ]
    for e in sorted(cuda_events, key=lambda x: -x.device_time_total):
        print(f"  {e.count:3d}x  {e.device_time_total / 1e3:8.2f} ms  {e.key[:90]}")
    total = sum(e.device_time_total for e in cuda_events) / 1e3
    print(f"  {'total':>4s}  {total:8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="MLP checkpoint benchmark")
    parser.add_argument("--B", type=int, default=4, help="batch size")
    parser.add_argument("--S", type=int, default=512, help="sequence length")
    parser.add_argument("--D", type=int, default=4096, help="model dimension")
    parser.add_argument("--hidden", type=int, default=None, help="hidden dim (default: 4*D)")
    parser.add_argument("--layers", type=int, default=4, help="layers for memory profiling")
    parser.add_argument("--activation", default="gelu_tanh_approx",
                        choices=["gelu_tanh_approx", "swiglu"])
    parser.add_argument("--no-memory", action="store_true", help="skip memory profiling")
    parser.add_argument("--no-kernels", action="store_true", help="skip kernel profiling")
    args = parser.parse_args()
    hidden = args.hidden or 4 * args.D

    # Memory profiling
    if not args.no_memory:
        S_mem = 2048  # larger seq len for memory profiling
        input_sz = args.B * S_mem * args.D * 2 / 1e6
        preact_sz = args.B * S_mem * hidden * 2 / 1e6
        print(f"Input: {input_sz:.0f} MB, Preact per layer: {preact_sz:.0f} MB")
        print(f"Without checkpoint: ~{args.layers * preact_sz:.0f} MB of saved activations")

        mem_results = {}
        for mode in ["normal", "checkpoint", "recompute"]:
            mem_results[mode] = run_memory(
                mode, args.B, S_mem, args.D, hidden, args.layers, args.activation
            )

        print(f"\n{'='*65}")
        print(f"  {'':20s}  {'normal':>10s}  {'checkpoint':>10s}  {'recompute':>10s}")
        print(f"  {'activation memory':20s}  {mem_results['normal'][0]:>8.0f} MB"
              f"  {mem_results['checkpoint'][0]:>8.0f} MB  {mem_results['recompute'][0]:>8.0f} MB")
        print(f"  {'peak memory':20s}  {mem_results['normal'][1]:>8.0f} MB"
              f"  {mem_results['checkpoint'][1]:>8.0f} MB  {mem_results['recompute'][1]:>8.0f} MB")
        print(f"{'='*65}")

    # Kernel profiling
    if not args.no_kernels:
        for act in ([args.activation] if args.activation != "all"
                    else ["gelu_tanh_approx", "swiglu"]):
            for mode in MODES:
                events = run_kernel_profile(mode, args.B, args.S, args.D, hidden, act)
                print(f"\n{'='*80}")
                print(f"  {act}, {mode}")
                print(f"{'='*80}")
                print_kernel_events(events)


if __name__ == "__main__":
    main()
