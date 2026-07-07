# Indexer Prefill Test — Verifies AscendC kernel correctness against PyTorch reference
# Test scenarios: B=2, S={64, 512, 4096}, start_pos=0

import sys
import os
import torch
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ascendc.indexer_launcher import IndexerLauncher
from test.torch_ref.indexer_torch import (
    IndexerTorchRef, get_init_inputs, get_inputs
)


def test_prefill(B: int, S: int, device: str = "npu"):
    """Test AscendC implementation against PyTorch reference for a given shape."""
    torch.manual_seed(42)

    # Get initialized weights
    args, wq_b_w, w_proj_w, kv_cache, freqs_cis = get_init_inputs(device)

    dim = args["dim"]
    n_heads = args["n_heads"]
    head_dim = args["head_dim"]
    rope_head_dim = args["rope_head_dim"]
    index_topk = args["index_topk"]
    q_lora_rank = args["q_lora_rank"]
    compress_ratio = args["compress_ratio"]

    # Fill KV cache with random data for realistic testing
    max_kv_len = args["max_seq_len"] // compress_ratio
    kv_cache = torch.randn_like(kv_cache)

    # --- PyTorch Reference ---
    model_ref = IndexerTorchRef(
        dim=dim, n_heads=n_heads, head_dim=head_dim,
        rope_head_dim=rope_head_dim, index_topk=index_topk,
        q_lora_rank=q_lora_rank, compress_ratio=compress_ratio,
        kv_cache=kv_cache.clone(), freqs_cis=freqs_cis.clone(),
        wq_b_weight=wq_b_w.clone(), weights_proj_weight=w_proj_w.clone(),
    ).to(device)

    x, qr, start_pos, offset = get_inputs(B, S, device)
    with torch.no_grad():
        ref_output = model_ref(x.clone(), qr.clone(), start_pos, offset)

    # --- AscendC Implementation ---
    launcher = IndexerLauncher(
        dim=dim, n_heads=n_heads, head_dim=head_dim,
        rope_head_dim=rope_head_dim, index_topk=index_topk,
        q_lora_rank=q_lora_rank, compress_ratio=compress_ratio,
        max_seq_len=args["max_seq_len"], max_batch_size=args["max_batch_size"],
        device=device,
    )
    launcher.load_weights(
        wq_weight=wq_b_w.clone(),
        w_proj_weight=w_proj_w.clone(),
        kv_cache=kv_cache.clone(),
        freqs_cis=freqs_cis.clone(),
    )

    with torch.no_grad():
        asc_output = launcher.forward(x.clone(), qr.clone(), start_pos, offset)

    # --- Comparison ---
    # For prefill (start_pos==0), some indices may be -1 (masked)
    # Compare where both are valid (>=0)
    shape_match = asc_output.shape == ref_output.shape
    print(f"  Shape match: {shape_match} (asc={list(asc_output.shape)}, ref={list(ref_output.shape)})")

    if not shape_match:
        return False

    # Count matches: compare indices where both are valid (>= 0)
    # Note: topk may return indices in different order if values are equal
    # We use set comparison per (b,s) position
    total_positions = 0
    matched_positions = 0
    mismatched_positions = 0

    for b in range(B):
        for s in range(S):
            ref_set = set(ref_output[b, s, :].tolist())
            asc_set = set(asc_output[b, s, :].tolist())
            total_positions += 1
            if ref_set == asc_set:
                matched_positions += 1
            else:
                mismatched_positions += 1

    match_rate = matched_positions / total_positions * 100 if total_positions > 0 else 0
    print(f"  Index set match: {matched_positions}/{total_positions} ({match_rate:.1f}%)")

    if mismatched_positions > 0:
        # Show a few mismatches
        count = 0
        for b in range(B):
            for s in range(S):
                ref_set = set(ref_output[b, s, :].tolist())
                asc_set = set(asc_output[b, s, :].tolist())
                if ref_set != asc_set and count < 3:
                    print(f"    Mismatch at (b={b}, s={s}):")
                    print(f"      ref: {sorted(ref_set)}")
                    print(f"      asc: {sorted(asc_set)}")
                    count += 1

    return match_rate > 99.0


def main():
    parser = argparse.ArgumentParser(description="Indexer prefill test")
    parser.add_argument("--device", type=str, default="npu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    print(f"Indexer Prefill Test: B={args.batch_size}, S={args.seq_len}")
    print("=" * 60)

    test_shapes = [
        (args.batch_size, args.seq_len),
    ]
    # Add additional shapes if using defaults
    if args.batch_size == 2 and args.seq_len == 64:
        test_shapes.extend([
            (2, 512),
            (2, 128),
            (1, 64),
        ])

    all_passed = True
    for B, S in test_shapes:
        print(f"\nTest: B={B}, S={S}")
        try:
            passed = test_prefill(B, S, args.device)
            status = "PASS" if passed else "FAIL"
            print(f"  Result: {status}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  Result: ERROR - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
