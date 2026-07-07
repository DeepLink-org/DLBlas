# ----------------------------------------------------------------------------
# 结果验证 - act_quant_kernel
# ----------------------------------------------------------------------------

import numpy as np
import sys


def decode_fp8_vec(uint8_arr):
    """Decode fp8_e4m3fn uint8 array to fp32 values."""
    result = np.zeros(uint8_arr.shape, dtype=np.float32)
    for i in range(uint8_arr.size):
        v = int(uint8_arr.flat[i])
        sign = (v >> 7) & 1
        exp  = (v >> 3) & 0xF
        mant = v & 0x7
        if exp == 0:
            val = (1 - 2 * sign) * (mant / 8.0) * (2.0 ** (-6))
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = (1 - 2 * sign) * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        result.flat[i] = np.float32(val)
    return result


def verify_result(output_q_path, output_s_path, golden_q_path, golden_s_path):
    # Read outputs
    q_out = np.fromfile(output_q_path, dtype=np.uint8)
    s_out = np.fromfile(output_s_path, dtype=np.float32)
    q_golden = np.fromfile(golden_q_path, dtype=np.uint8)
    s_golden = np.fromfile(golden_s_path, dtype=np.float32)

    q_passed = True
    s_passed = True

    # Verify x_q (fp8 as uint8) - allow 1-ULP difference
    if q_out.shape != q_golden.shape:
        print(f"x_q shape mismatch: output {q_out.shape} vs golden {q_golden.shape}")
        q_passed = False
    else:
        q_diff = np.abs(q_out.astype(np.int32) - q_golden.astype(np.int32))
        q_diff_exact = np.sum(q_diff != 0)
        q_diff_big = np.sum(q_diff > 1)  # more than 1 ULP
        if q_diff_big == 0:
            if q_diff_exact == 0:
                print(f"x_q (fp8) verification PASSED! Shape: {q_out.shape}, exact match")
            else:
                print(f"x_q (fp8) verification PASSED! Shape: {q_out.shape}, "
                      f"{q_diff_exact}/{q_out.size} elements with 1-ULP diff (acceptable)")
            q_passed = True
        else:
            print(f"x_q (fp8) verification FAILED! >1 ULP mismatches: {q_diff_big}/{q_out.size}")
            bad_idx = np.where(q_diff > 1)[0][:10]
            for idx in bad_idx:
                print(f"  [{idx}] out=0x{q_out[idx]:02X} golden=0x{q_golden[idx]:02X}")
            q_passed = False

    # Verify x_s (fp32 scale)
    if s_out.shape != s_golden.shape:
        print(f"x_s shape mismatch: output {s_out.shape} vs golden {s_golden.shape}")
        s_passed = False
    else:
        s_diff = np.abs(s_out - s_golden)
        s_max_diff = np.max(s_diff)
        s_rtol = 1e-5
        s_atol = 1e-6
        s_close = np.allclose(s_out, s_golden, rtol=s_rtol, atol=s_atol)
        if s_close:
            print(f"x_s (fp32) verification PASSED! Shape: {s_out.shape}, max_diff={s_max_diff:.2e}")
        else:
            print(f"x_s (fp32) verification FAILED! max_diff={s_max_diff:.2e}")
            print(f"  Sample: out={s_out[:5]}")
            print(f"  Sample: gold={s_golden[:5]}")
            s_passed = False

    return q_passed and s_passed


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python verify_result.py <output_q.bin> <output_s.bin> <golden_q.bin> <golden_s.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0 if success else 1)
