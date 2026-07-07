# --------------------------------------------------------------------------
# Multi-level test data generation for engram_gate_fwd
# --------------------------------------------------------------------------
# Generates binary input files and config for multiple test levels.
# Level 0: 8-16 elements (basic functionality)
# Level 1: ~1K elements (typical scenario)
# Level 2: Edge cases (single/zero values)
# --------------------------------------------------------------------------

import numpy as np
import os
import struct
import sys
import json

from golden import compute_golden, fp32_to_bf16, bf16_to_fp32


def generate_case(case_name, num_tokens, hc_mult, hidden_size, clamp_value=1e-6, eps=1e-20, seed=42):
    """Generate a single test case and run kernel verification."""
    np.random.seed(seed)

    # Generate input data in fp32 then convert to bf16
    hs_f32 = np.random.randn(num_tokens, hc_mult, hidden_size).astype(np.float32)
    hs_bf16 = fp32_to_bf16(hs_f32)

    k_f32 = np.random.randn(num_tokens, hc_mult, hidden_size).astype(np.float32)
    k_bf16 = fp32_to_bf16(k_f32)

    v_f32 = np.random.randn(num_tokens, hidden_size).astype(np.float32)
    v_bf16 = fp32_to_bf16(v_f32)

    wh_f32 = np.random.randn(hc_mult, hidden_size).astype(np.float32)
    wh_bf16 = fp32_to_bf16(wh_f32)

    we_f32 = np.random.randn(hc_mult, hidden_size).astype(np.float32)
    we_bf16 = fp32_to_bf16(we_f32)

    # Write bf16 input files
    os.makedirs("input", exist_ok=True)
    hs_bf16.tofile("input/input_hidden_states.bin")
    k_bf16.tofile("input/input_k.bin")
    v_bf16.tofile("input/input_v.bin")
    wh_bf16.tofile("input/input_weight_hidden.bin")
    we_bf16.tofile("input/input_weight_embed.bin")

    # Write config file
    config_bytes = struct.pack("QQQff", num_tokens, hc_mult, hidden_size, clamp_value, eps)
    with open("input/config.bin", "wb") as f:
        f.write(config_bytes)

    # Compute golden outputs
    hs_ref = bf16_to_fp32(hs_bf16).reshape(num_tokens, hc_mult, hidden_size)
    k_ref  = bf16_to_fp32(k_bf16).reshape(num_tokens, hc_mult, hidden_size)
    v_ref  = bf16_to_fp32(v_bf16).reshape(num_tokens, hidden_size)
    wh_ref = bf16_to_fp32(wh_bf16).reshape(hc_mult, hidden_size)
    we_ref = bf16_to_fp32(we_bf16).reshape(hc_mult, hidden_size)

    output_f32, raw_dot, gate_score, rstd_x, rstd_k = compute_golden(
        hs_ref, k_ref, v_ref, wh_ref, we_ref,
        clamp_value, eps, hidden_size
    )

    os.makedirs("output", exist_ok=True)

    # output: bf16
    out_bf16 = fp32_to_bf16(output_f32)
    out_bf16.tofile("output/golden.bin")

    # scalar outputs: fp32
    raw_dot.astype(np.float32).tofile("output/golden_raw_dot.bin")
    gate_score.astype(np.float32).tofile("output/golden_gate_score.bin")
    rstd_x.astype(np.float32).tofile("output/golden_rstd_x.bin")
    rstd_k.astype(np.float32).tofile("output/golden_rstd_k.bin")

    return {
        "name": case_name,
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
        "clamp_value": clamp_value,
        "eps": eps
    }


# ============================================================================
# Test case definitions
# ============================================================================

TEST_CASES = [
    # Level 0: Small scale basic verification (8-16 element range)
    {"name": "L0_small_basic",     "nt": 2,  "hc": 2, "hs": 16},
    {"name": "L0_small_hs256",     "nt": 2,  "hc": 2, "hs": 256},

    # Level 1: Typical scenario (~1K elements)
    {"name": "L1_typical",         "nt": 32, "hc": 4, "hs": 1024},
    {"name": "L1_large",           "nt": 32, "hc": 4, "hs": 4096},

    # Level 2: Edge cases
    {"name": "L2_single_token",    "nt": 1,  "hc": 4, "hs": 4096},
    {"name": "L2_single_hc",       "nt": 8,  "hc": 1, "hs": 4096},
    {"name": "L2_small_hs",        "nt": 8,  "hc": 4, "hs": 512},
    {"name": "L2_unaligned",       "nt": 4,  "hc": 4, "hs": 4097},
    {"name": "L2_large_hs",        "nt": 4,  "hc": 4, "hs": 8192},
]


def main():
    if len(sys.argv) > 1:
        # Run a specific case by index
        idx = int(sys.argv[1])
        case = TEST_CASES[idx]
        info = generate_case(case["name"], case["nt"], case["hc"], case["hs"])
        print(f"Generated test case [{idx}]: {info['name']}")
        print(f"  Shape: num_tokens={info['num_tokens']}, hc_mult={info['hc_mult']}, hidden_size={info['hidden_size']}")
        # Also write case index to a file for verification script
        with open("input/case_index.txt", "w") as f:
            f.write(str(idx))
    else:
        # Default: generate all cases sequentially and run kernel for each
        # But for practical use in run.sh, generate the default profiling case
        info = generate_case(TEST_CASES[3]["name"], TEST_CASES[3]["nt"],
                           TEST_CASES[3]["hc"], TEST_CASES[3]["hs"])
        print(f"Generated test data:")
        print(f"  Shape: num_tokens={info['num_tokens']}, hc_mult={info['hc_mult']}, hidden_size={info['hidden_size']}")
        print(f"  clamp_value={info['clamp_value']}, eps={info['eps']}")
        print(f"  Input files: input/input_*.bin")
        print(f"  Config: input/config.bin")
        print(f"  Golden files: output/golden*.bin")

    return 0


if __name__ == "__main__":
    sys.exit(main())
