import torch
import time
import numpy as np

def head_compute_mix_bwd_torch(input_mix, mhc_scale, mhc_base, grad_out):
    """PyTorch reference implementation of head_compute_mix_bwd."""
    z = input_mix * mhc_scale + mhc_base
    sigmoid = torch.sigmoid(z)
    grad_z = grad_out * sigmoid * (1 - sigmoid)
    grad_input_mix = grad_z * mhc_scale
    grad_mhc_base = grad_z.sum(dim=(0, 1)).view(-1)
    grad_mhc_scale = (grad_z * input_mix).sum(dim=(0, 1, 2)).view(1)
    return grad_input_mix, grad_mhc_scale, grad_mhc_base

def main():
    batch0, batch1, mhc_mult = 2, 1024, 4
    warmup = 10
    iterations = 100

    # Use GPU if available (MACA device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create inputs
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device=device)
    mhc_scale = torch.randn(1, dtype=torch.float32, device=device)
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32, device=device)
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup):
        _ = head_compute_mix_bwd_torch(input_mix, mhc_scale, mhc_base, grad_out)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = head_compute_mix_bwd_torch(input_mix, mhc_scale, mhc_base, grad_out)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000.0
    avg_time_ms = total_time_ms / iterations

    print(f"PyTorch head_compute_mix_bwd:")
    print(f"  Total time ({iterations} iters): {total_time_ms:.6f} ms")
    print(f"  Average time: {avg_time_ms:.6f} ms")
    print(f"  Shape: ({batch0}, {batch1}, {mhc_mult})")

    # Also benchmark individual ops for breakdown
    # Fused sigmoid_backward equivalent
    input_mix_np = input_mix.cpu().numpy()
    mhc_scale_np = mhc_scale.cpu().numpy()
    mhc_base_np = mhc_base.cpu().numpy()
    grad_out_np = grad_out.cpu().numpy()

    print(f"\n  Input shapes: input_mix={input_mix.shape}, mhc_scale={mhc_scale.shape}, "
          f"mhc_base={mhc_base.shape}, grad_out={grad_out.shape}")
    print(f"  Total elements: {batch0 * batch1 * mhc_mult}")

    return avg_time_ms

if __name__ == "__main__":
    torch_time = main()
    print(f"\n<torch_time>{torch_time:.6f} ms</torch_time>")
