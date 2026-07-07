# --------------------------------------------------------------------------
# Verification for engram_gate_fwd direct invoke
# --------------------------------------------------------------------------
# Compares kernel output binaries against golden reference.
# --------------------------------------------------------------------------

import numpy as np
import os
import struct
import sys

from golden import bf16_to_fp32


def load_bf16(filename):
    """Load bf16 file (uint16) and convert to fp32."""
    raw = np.fromfile(filename, dtype=np.uint16)
    return bf16_to_fp32(raw)


def load_fp32(filename):
    """Load fp32 file."""
    return np.fromfile(filename, dtype=np.float32)


def main():
    # Read config to get shapes
    with open("input/config.bin", "rb") as f:
        cfg_bytes = f.read(32)
    num_tokens, hc_mult, hidden_size = struct.unpack_from("QQQ", cfg_bytes, 0)
    total_rows = num_tokens * hc_mult

    print(f"Verification: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")

    # ================================================================
    # Output (bf16) verification
    # ================================================================
    print("\n--- Output (bf16) ---")
    kernel_out = load_bf16("output/output.bin")
    golden_out = load_bf16("output/golden.bin")

    if kernel_out.shape != golden_out.shape:
        print(f"  ERROR: shape mismatch: kernel={kernel_out.shape}, golden={golden_out.shape}")
        sys.exit(1)

    kernel_out = kernel_out.reshape(num_tokens, hc_mult, hidden_size)
    golden_out = golden_out.reshape(num_tokens, hc_mult, hidden_size)

    abs_err = np.abs(kernel_out - golden_out)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)

    rel_err = abs_err / (np.abs(golden_out) + 1e-8)
    max_rel_err = np.max(rel_err)
    mean_rel_err = np.mean(rel_err)

    print(f"  Max abs error:  {max_abs_err:.6e}")
    print(f"  Mean abs error: {mean_abs_err:.6e}")
    print(f"  Max rel error:  {max_rel_err:.6e}")
    print(f"  Mean rel error: {mean_rel_err:.6e}")

    bf16_pass = max_rel_err < 1e-3 or max_abs_err < 1e-2

    # ================================================================
    # Scalar outputs (fp32) verification
    # ================================================================
    scalar_outputs = {
        "raw_dot":    ("output/output_raw_dot.bin", "output/golden_raw_dot.bin"),
        "gate_score": ("output/output_gate_score.bin", "output/golden_gate_score.bin"),
        "rstd_x":     ("output/output_rstd_x.bin", "output/golden_rstd_x.bin"),
        "rstd_k":     ("output/output_rstd_k.bin", "output/golden_rstd_k.bin"),
    }

    all_pass = bf16_pass

    for name, (kernel_file, golden_file) in scalar_outputs.items():
        print(f"\n--- {name} (fp32) ---")
        k_val = load_fp32(kernel_file)
        g_val = load_fp32(golden_file)

        if k_val.shape != g_val.shape:
            print(f"  ERROR: shape mismatch: kernel={k_val.shape}, golden={g_val.shape}")
            all_pass = False
            continue

        abs_err = np.abs(k_val - g_val)
        max_ae = np.max(abs_err)
        mean_ae = np.mean(abs_err)

        rel_err = abs_err / (np.abs(g_val) + 1e-8)
        max_re = np.max(rel_err)
        mean_re = np.mean(rel_err)

        print(f"  Max abs error:  {max_ae:.6e}")
        print(f"  Mean abs error: {mean_ae:.6e}")
        print(f"  Max rel error:  {max_re:.6e}")
        print(f"  Mean rel error: {mean_re:.6e}")

        scalar_pass = max_re < 1e-4 or max_ae < 1e-5
        print(f"  {'PASS' if scalar_pass else 'FAIL'}")
        all_pass = all_pass and scalar_pass

    print(f"\n{'='*60}")
    if all_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print(f"{'='*60}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
