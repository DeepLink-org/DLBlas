# Indexer Decode Test — Verifies AscendC kernel correctness for decode phase
# Test scenarios: B=2, S=1, start_pos={64, 512}

import sys
import os
import torch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ascendc.indexer_launcher import IndexerLauncher
from test.torch_ref.indexer_torch import (
    IndexerTorchRef, get_init_inputs
)


def test_decode(B: int, start_pos: int, offset: int = 0, device: str = "npu"):
    """Test AscendC implementation against PyTorch reference for decode phase."""
    torch.manual_seed(123)

    args, wq_b_w, w_proj_w, kv_cache, freqs_cis = get_init_inputs(device)

    dim = args["dim"]
    n_heads = args["n_heads"]
    head_dim = args["head_dim"]
    rope_head_dim = args["rope_head_dim"]
    index_topk = args["index_topk"]
    q_lora_rank = args["q_lora_rank"]
    compress_ratio = args["compress_ratio"]

    # Fill KV cache with random data
    kv_cache = torch.randn_like(kv_cache)

    # Decode: S=1
    S = 1
    x = torch.randn(B, S, dim, dtype=torch.bfloat16, device=device)
    qr = torch.randn(B, S, q_lora_rank, dtype=torch.bfloat16, device=device)

    # --- PyTorch Reference ---
    model_ref = IndexerTorchRef(
        dim=dim, n_heads=n_heads, head_dim=head_dim,
        rope_head_dim=rope_head_dim, index_topk=index_topk,
        q_lora_rank=q_lora_rank, compress_ratio=compress_ratio,
        kv_cache=kv_cache.clone(), freqs_cis=freqs_cis.clone(),
        wq_b_weight=wq_b_w.clone(), weights_proj_weight=w_proj_w.clone(),
    ).to(device)

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
    shape_match = asc_output.shape == ref_output.shape
    print(f"  Shape match: {shape_match} (asc={list(asc_output.shape)}, ref={list(ref_output.shape)})")

    if not shape_match:
        return False

    # For decode (start_pos > 0), no causal mask — all indices are valid.
    # Use set comparison per (b,s) position: topk order is undefined for equal-valued
    # indices, and fp32 (Triton kernel) vs bf16 (reference) accumulation causes
    # different rounding → different ordering of tied scores. Set comparison
    # correctly verifies that the same top-k indices were selected regardless of order.
    total_positions = 0
    matched_positions = 0
    mismatched_positions = 0

    B, S, K = asc_output.shape
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
        # Show mismatches with details
        count = 0
        for b in range(B):
            for s in range(S):
                ref_set = set(ref_output[b, s, :].tolist())
                asc_set = set(asc_output[b, s, :].tolist())
                if ref_set != asc_set and count < 3:
                    print(f"    Mismatch at (b={b}, s={s}):")
                    print(f"      ref: {sorted(ref_set)}")
                    print(f"      asc: {sorted(asc_set)}")
                    print(f"      ref-only: {sorted(ref_set - asc_set)}")
                    print(f"      asc-only: {sorted(asc_set - ref_set)}")
                    count += 1

    return match_rate > 99.0


def main():
    parser = argparse.ArgumentParser(description="Indexer decode test")
    parser.add_argument("--device", type=str, default="npu")
    args = parser.parse_args()

    print("Indexer Decode Test (S=1)")
    print("=" * 60)

    test_cases = [
        (2, 64, 0),
        (2, 128, 0),
        (1, 512, 100),
    ]

    all_passed = True
    for B, start_pos, offset in test_cases:
        print(f"\nTest: B={B}, start_pos={start_pos}, offset={offset}")
        try:
            passed = test_decode(B, start_pos, offset, args.device)
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
