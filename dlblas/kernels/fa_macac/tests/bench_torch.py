import math, torch
from flash_attn.flash_attn_interface import _flash_attn_forward

torch.manual_seed(0)
D = 128; H = 32; B = 1
scale = 1.0 / math.sqrt(D)
shapes = [512, 1024, 10240, 102400]
warmup = 5
iters = 30

print(f"{'S':>8} {'torch_ms':>12}")
for S in shapes:
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    # wheel expects B,H,S,D -> transpose to (B,H_head... ) flash layout (B, nheads, seqlen, headdim) same here
    qf = q.transpose(1, 2).contiguous()
    kf = k.transpose(1, 2).contiguous()
    vf = v.transpose(1, 2).contiguous()

    # warmup
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
    ms = start.elapsed_time(end) / iters
    print(f"{S:>8} {ms:>12.6f}")
