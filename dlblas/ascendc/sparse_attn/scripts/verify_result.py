# ----------------------------------------------------------------------------------------------------------
# verify_result.py - Precision verification for sparse_attn (proper bf16 handling)
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import sys

rtol = 1e-2
atol = 1e-2

def verify(output_path, golden_path):
    # Read bf16 data: stored as int16 raw bytes
    out_raw = np.fromfile(output_path, dtype=np.int16)
    gld_raw = np.fromfile(golden_path, dtype=np.int16)

    if out_raw.shape != gld_raw.shape:
        print(f"Shape mismatch: output {out_raw.shape} vs golden {gld_raw.shape}")
        return False

    # Convert bf16 uint16 → fp32 for comparison
    out_f32 = (out_raw.astype(np.uint32) << 16).view(np.float32)
    gld_f32 = (gld_raw.astype(np.uint32) << 16).view(np.float32)

    diff = np.abs(out_f32 - gld_f32)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)

    passed = np.allclose(out_f32, gld_f32, rtol=rtol, atol=atol, equal_nan=True)

    if passed:
        print(f"PASSED! Shape: {out_raw.shape}, Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        return True
    else:
        print(f"FAILED! Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        m = diff > (atol + rtol * np.abs(gld_f32))
        print(f"Mismatch: {m.sum()}/{len(gld_f32)}")
        if m.sum()>0:
            mi = np.where(m)[0][:5]
            for i in mi:
                print(f"  [{i}] out={out_f32[i]:.6f} gld={gld_f32[i]:.6f} diff={diff[i]:.6e}")
        return False

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>"); sys.exit(1)
    sys.exit(0 if verify(sys.argv[1], sys.argv[2]) else 1)
