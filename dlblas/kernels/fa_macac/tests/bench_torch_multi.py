import math, statistics, torch
from flash_attn.flash_attn_interface import _flash_attn_forward

torch.manual_seed(0)
D = 128; H = 32; B = 1
scale = 1.0 / math.sqrt(D)
shapes = [512, 1024, 10240]
runs = 5
warmup = 5
iters = 30

for S in shapes:
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    qf = q.transpose(1, 2).contiguous()
    kf = k.transpose(1, 2).contiguous()
    vf = v.transpose(1, 2).contiguous()
    samples = []
    for _ in range(runs):
        for _ in range(warmup):
            _flash_attn_forward(qf, kf, vf, 0.0, scale, False, (-1, -1))
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _flash_attn_forward(qf, kf, vf, 0.0, scale, False, (-1, -1))
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    print(f"S={S:>6}: min={min(samples):.6f} med={statistics.median(samples):.6f} max={max(samples):.6f} ms  (n={runs})")
