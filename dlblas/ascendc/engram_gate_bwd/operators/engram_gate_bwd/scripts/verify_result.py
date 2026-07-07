# ----------------------------------------------------------------------------------------------------------
# Result verification for engram_gate_bwd (bf16 output vs f32 golden)
# ----------------------------------------------------------------------------------------------------------

import numpy as np
import sys
import os

rtol = 1e-2   # bf16 precision
atol = 1e-3

OUTPUT_NAMES = ["grad_x", "grad_k", "grad_v", "grad_wh", "grad_we"]


def load_bf16(path, shape_flat):
    """Load bf16 packed data (uint16) and convert to f32"""
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != shape_flat:
        print(f"  Size mismatch: got {raw.size}, expected {shape_flat}")
        return None
    # bf16 → f32: shift left by 16
    f32 = raw.astype(np.uint32) << 16
    return f32.view(np.float32)


def verify_one(out_path, gold_path, name, shape_flat):
    golden = np.fromfile(gold_path, dtype=np.float32)
    output = load_bf16(out_path, shape_flat)

    if output is None:
        return False

    if output.shape != golden.shape:
        print(f"  [{name}] SHAPE MISMATCH: output {output.shape} vs golden {golden.shape}")
        return False

    diff = np.abs(output - golden)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_err = diff / (np.abs(golden) + 1e-8)
    max_rel_err = np.max(rel_err)

    passed = np.allclose(output, golden, rtol=rtol, atol=atol)

    if passed:
        print(f"  [{name}] PASS max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} max_rel={max_rel_err:.6e} shape={output.shape}")
    else:
        print(f"  [{name}] FAIL max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} max_rel={max_rel_err:.6e} shape={output.shape}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"           mismatches: {len(mismatches)}/{len(golden)}")

    return passed


def verify_all():
    shapes = {
        "grad_x": 14*4*128,
        "grad_k": 14*4*128,
        "grad_v": 14*128,
        "grad_wh": 4*128,
        "grad_we": 4*128,
    }

    all_passed = True
    for name in OUTPUT_NAMES:
        out_path = f"output/output_{name}.bin"
        gold_path = f"output/golden_{name}.bin"
        if not os.path.exists(out_path) or not os.path.exists(gold_path):
            print(f"  [{name}] SKIP - file missing")
            all_passed = False
            continue
        if not verify_one(out_path, gold_path, name, shapes[name]):
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print(f"Verification: rtol={rtol}, atol={atol}")
    success = verify_all()
    print("ALL PASSED" if success else "SOME FAILED")
    sys.exit(0 if success else 1)
