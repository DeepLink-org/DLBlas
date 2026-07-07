# --------------------------------------------------------------------------
# PyTorch pathway test for engram_gate_fwd
# --------------------------------------------------------------------------

import sys
import os

import torch
import torch_npu

from golden import compute_golden, fp32_to_bf16, bf16_to_fp32

SO_NAME  = "libengram_gate_fwd_ops.so"
OP_NAME  = "engram_gate_fwd"
ATOL     = 1e-2   # bf16 tolerance
RTOL     = 1e-3


def fp32_to_bf16_torch(x):
    """Convert fp32 tensor to bf16 tensor (using torch bfloat16)."""
    return x.to(torch.bfloat16)


def run_test(name, num_tokens, hc_mult, hidden_size, clamp_value, eps):
    """Run single test case, return (name, passed, messages)."""
    op_fn = getattr(torch.ops.npu, OP_NAME)

    # Generate test data
    torch.manual_seed(42)
    hs = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    k  = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    v  = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    wh = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
    we = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)

    # Run kernel
    try:
        result = op_fn(hs.npu(), k.npu(), v.npu(),
                       wh.npu(), we.npu(),
                       clamp_value, eps)
        out_kernel, rd_kernel, gs_kernel, rx_kernel, rk_kernel = result
    except Exception as e:
        return name, False, f"Kernel call failed: {e}"

    # Compute golden in fp32
    hs_f32 = hs.float()
    k_f32  = k.float()
    v_f32  = v.float()
    wh_f32 = wh.float()
    we_f32 = we.float()

    golden_out_f32, golden_rd, golden_gs, golden_rx, golden_rk = compute_golden(
        hs_f32.cpu().numpy(), k_f32.cpu().numpy(), v_f32.cpu().numpy(),
        wh_f32.cpu().numpy(), we_f32.cpu().numpy(),
        clamp_value, eps, hidden_size)

    golden_out = torch.from_numpy(golden_out_f32).bfloat16().npu()
    golden_rd  = torch.from_numpy(golden_rd).npu()
    golden_gs  = torch.from_numpy(golden_gs).npu()
    golden_rx  = torch.from_numpy(golden_rx).npu()
    golden_rk  = torch.from_numpy(golden_rk).npu()

    # Check output (bf16)
    out_diff = torch.max(torch.abs(out_kernel.float() - golden_out.float())).item()
    out_pass = out_diff < 1e-1  # bf16 can have larger absolute diff

    # Check scalar outputs (fp32)
    rd_diff = torch.max(torch.abs(rd_kernel - golden_rd)).item()
    rd_pass = rd_diff < 1e-4 or torch.allclose(rd_kernel, golden_rd, rtol=1e-3)

    gs_diff = torch.max(torch.abs(gs_kernel - golden_gs)).item()
    gs_pass = gs_diff < 1e-4 or torch.allclose(gs_kernel, golden_gs, rtol=1e-3)

    rx_diff = torch.max(torch.abs(rx_kernel - golden_rx)).item()
    rx_pass = rx_diff < 1e-4 or torch.allclose(rx_kernel, golden_rx, rtol=1e-3)

    rk_diff = torch.max(torch.abs(rk_kernel - golden_rk)).item()
    rk_pass = rk_diff < 1e-4 or torch.allclose(rk_kernel, golden_rk, rtol=1e-3)

    all_pass = out_pass and rd_pass and gs_pass and rx_pass and rk_pass
    msgs = (f"out_diff={out_diff:.6f} rd_diff={rd_diff:.6f} "
            f"gs_diff={gs_diff:.6f} rx_diff={rx_diff:.6f} rk_diff={rk_diff:.6f}")
    return name, all_pass, msgs


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"NOTE: {so_path} not found. Skipping PyTorch test.")
        print(f"Build the torch target first: cmake .. && make")
        sys.exit(0)
    torch.ops.load_library(so_path)

    results = []

    # Level 0: Small smoke test
    results.append(run_test("L0_small", num_tokens=2, hc_mult=2,
                             hidden_size=256, clamp_value=1e-6, eps=1e-20))

    # Level 1: Medium test
    results.append(run_test("L1_medium", num_tokens=8, hc_mult=4,
                             hidden_size=1024, clamp_value=1e-6, eps=1e-20))

    total   = len(results)
    passed  = sum(r[1] for r in results)
    failed  = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, msg in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'} ({msg})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
