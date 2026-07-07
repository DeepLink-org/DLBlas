# ============================================================================
# engram_gate_w_reduce PyTorch 通路测试脚本
# ============================================================================

import sys
import os
import argparse

import torch
import torch_npu
import numpy as np

from golden import compute_golden, bf16_to_fp32

SO_NAME = "libengram_gate_w_reduce_ops.so"
OP_NAME = "engram_gate_w_reduce"
ATOL = 1e-5
RTOL = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=4096)
args = parser.parse_args()


def to_bfloat16_tensor(arr_fp32):
    """Convert FP32 torch tensor to BF16 (uint16 representation) using numpy."""
    arr_np = arr_fp32.numpy()
    arr_u32 = arr_np.view(np.uint32)
    arr_bf16 = (arr_u32 >> 16).astype(np.uint16)
    return torch.from_numpy(arr_bf16)


def run_test(hidden_size):
    """Run PyTorch integration test."""
    R, C, H = 108, 4, hidden_size

    # Generate test data on CPU then move to NPU
    grad_w_partial = torch.randn(R, C, H, dtype=torch.float32)
    weight_hidden_fp32 = torch.randn(C, H, dtype=torch.float32)
    weight_embed_fp32 = torch.randn(C, H, dtype=torch.float32)
    grad_wh_in = torch.randn(C, H, dtype=torch.float32)
    grad_we_in = torch.randn(C, H, dtype=torch.float32)

    # Compute golden on CPU (FP32 full precision)
    gold_hidden, gold_embed = compute_golden(
        grad_w_partial.numpy(),
        weight_hidden_fp32.numpy(),
        weight_embed_fp32.numpy(),
        grad_wh_in.numpy(),
        grad_we_in.numpy()
    )

    # Convert weights to BF16
    weight_hidden_bf16 = to_bfloat16_tensor(weight_hidden_fp32)
    weight_embed_bf16 = to_bfloat16_tensor(weight_embed_fp32)

    # Move to NPU
    grad_w_partial_npu = grad_w_partial.npu()
    weight_hidden_npu = weight_hidden_bf16.view(torch.bfloat16).npu()
    weight_embed_npu = weight_embed_bf16.view(torch.bfloat16).npu()
    grad_wh_in_npu = grad_wh_in.clone().npu()
    grad_we_in_npu = grad_we_in.clone().npu()

    # Call the Ascend C operator
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y_hidden, y_embed = op_fn(
        grad_w_partial_npu,
        weight_hidden_npu,
        weight_embed_npu,
        grad_wh_in_npu,
        grad_we_in_npu
    )

    # Move results back to CPU
    y_hidden_cpu = y_hidden.cpu()
    y_embed_cpu = y_embed.cpu()

    # Compare with golden
    gold_hidden_t = torch.from_numpy(gold_hidden)
    gold_embed_t = torch.from_numpy(gold_embed)

    diff_h = torch.abs(y_hidden_cpu - gold_hidden_t).max().item()
    diff_e = torch.abs(y_embed_cpu - gold_embed_t).max().item()

    ok_h = torch.allclose(y_hidden_cpu, gold_hidden_t, rtol=RTOL, atol=ATOL)
    ok_e = torch.allclose(y_embed_cpu, gold_embed_t, rtol=RTOL, atol=ATOL)
    passed = ok_h and ok_e

    return passed, diff_h, diff_e


def main():
    so_path = SO_NAME if os.path.exists(SO_NAME) else os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    H = args.hidden_size
    passed, diff_h, diff_e = run_test(H)

    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME}, hidden_size={H})")
    print(f"{'='*50}")
    print(f"  grad_weight_hidden: {'PASSED' if passed else 'FAILED'} (max_diff={diff_h:.6e})")
    print(f"  grad_weight_embed:  {'PASSED' if passed else 'FAILED'} (max_diff={diff_e:.6e})")
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    print(f"{'='*50}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
