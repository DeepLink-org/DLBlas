#!/usr/bin/env python3
"""
Requires: https://github.com/facebookexperimental/triton (TLX extensions)
"""

import math
import torch
import triton
import triton.language as tl
from triton.language.extra import tlx

import tilelang
import tilelang.language as T


def _bh_for(H: int) -> int:
    for b in (1024, 512, 256, 128, 64):
        if H % b == 0:
            return b
    return 32


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["N", "hidden_size"],
)
@triton.jit
def _k_fuse_tlx(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    N,
    hidden_size,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    hc_mult: tl.constexpr,
    hc_mult3: tl.constexpr,
    hidden_block: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    n_splits: tl.constexpr,
    num_hidden_blocks: tl.constexpr,
):
    pid = tl.program_id(0)

    mo = tl.arange(0, hc_mult)
    ri = mo[:, None]
    ci = mo[None, :]
    ho = tl.arange(0, hidden_block)
    rms_acc = 0.0
    for s in range(n_splits):
        rms_acc += tl.load(gemm_out_sqrsum + s * N + pid)

    inv_rms = tl.math.rsqrt(
        rms_acc / (hc_mult * hidden_size) + rms_eps
    )

    pre_mix_raw = tl.zeros([hc_mult], dtype=tl.float32)
    post_mix_raw = tl.zeros([hc_mult], dtype=tl.float32)
    comb_mix_raw = tl.zeros([hc_mult, hc_mult], dtype=tl.float32)

    for s in range(n_splits):
        base = s * N * hc_mult3 + pid * hc_mult3

        pre_mix_raw += tl.load(gemm_out_mul + base + mo)
        post_mix_raw += tl.load(gemm_out_mul + base + hc_mult + mo)
        comb_mix_raw += tl.load(
            gemm_out_mul + base + 2 * hc_mult + ri * hc_mult + ci
        )

    pre_mix_raw *= inv_rms
    post_mix_raw *= inv_rms
    comb_mix_raw *= inv_rms

  
    s1 = tl.load(hc_scale + 1)
    post_base = tl.load(hc_base + hc_mult + mo)

    post_out = (
        tl.sigmoid(post_mix_raw * s1 + post_base)
        * hc_post_mult_value
    )

    tl.store(post_mix + pid * hc_mult + mo, post_out)

   
    s2 = tl.load(hc_scale + 2)
    comb_base = tl.load(
        hc_base + 2 * hc_mult + ri * hc_mult + ci
    )

    comb_mat = comb_mix_raw * s2 + comb_base

    row_max = tl.max(comb_mat, axis=1)
    comb_mat = tl.exp(comb_mat - row_max[:, None])

    row_sum = tl.sum(comb_mat, axis=1)
    comb_mat = comb_mat / row_sum[:, None] + hc_sinkhorn_eps

    col_sum = tl.sum(comb_mat, axis=0)
    comb_mat = comb_mat / (
        col_sum[None, :] + hc_sinkhorn_eps
    )

    for _ in range(sinkhorn_repeat - 1):
        row_sum = tl.sum(comb_mat, axis=1)
        comb_mat = comb_mat / (
            row_sum[:, None] + hc_sinkhorn_eps
        )
        col_sum = tl.sum(comb_mat, axis=0)
        comb_mat = comb_mat / (
            col_sum[None, :] + hc_sinkhorn_eps
        )

    tl.store(
        comb_mix + pid * hc_mult * hc_mult + ri * hc_mult + ci,
        comb_mat,
    )

    s0 = tl.load(hc_scale + 0)
    pre_base = tl.load(hc_base + mo)

    pre_mix_weight = (
        tl.sigmoid(pre_mix_raw * s0 + pre_base)
        + hc_pre_eps
    )

  
    residual_bufs = tlx.local_alloc(
        (hc_mult, hidden_block),
        tl.bfloat16,
        2,
    )

    residual_base = (
        residual + pid * hc_mult * hidden_size
    )

    off_m = mo[:, None] * hidden_size
    off_h = ho[None, :]

    if num_hidden_blocks > 0:
        tlx.async_load(
            residual_base + off_m + 0 * hidden_block + off_h,
            residual_bufs[0],
        )
        tlx.async_load_commit_group()

    for ih in range(num_hidden_blocks):
        curr_idx = ih % 2
        next_ih = ih + 1
        next_idx = next_ih % 2

        tlx.async_load_wait_group(0)

        residual_tile = tlx.local_load(
            residual_bufs[curr_idx]
        ).to(tl.float32)

        if next_ih < num_hidden_blocks:
            tlx.async_load(
                residual_base
                + off_m
                + next_ih * hidden_block
                + off_h,
                residual_bufs[next_idx],
            )
            tlx.async_load_commit_group()

        layer_acc = tl.sum(
            pre_mix_weight[:, None] * residual_tile,
            axis=0,
        )

        tl.store(
            layer_input
            + pid * hidden_size
            + ih * hidden_block
            + ho,
            layer_acc.to(tl.bfloat16),
        )

def mhc_pre_big_fuse_triton(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits,
    hc_mult,
):
    N = residual.shape[0]
    M = hc_mult
    M3 = 2 * M + M * M
    H = hidden_size
    BH = _bh_for(H)
    NH = H // BH
    assert NH >= 2
    _k_fuse_tlx[(N,)](
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    N,
    hidden_size,
    rms_eps=rms_eps,
    hc_pre_eps=hc_pre_eps,
    hc_sinkhorn_eps=hc_sinkhorn_eps,
    hc_post_mult_value=hc_post_mult_value,
    hc_mult=M,
    hc_mult3=M3,
    hidden_block=BH,
    sinkhorn_repeat=sinkhorn_repeat,
    n_splits=n_splits,
    num_hidden_blocks=NH,
)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + hc_pre_eps
                )
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)
                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                T.copy(ol, layer_input[i, i0_h * hidden_block])


@torch.no_grad()
def big_fuse_ref_pytorch(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    hidden_size,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits,
    hc_mult,
):
    N = residual.shape[0]
    M = hc_mult
    sqrsum = gemm_out_sqrsum.sum(0)
    inv_rms = torch.rsqrt(sqrsum / (M * hidden_size) + rms_eps)
    mixes = gemm_out_mul.sum(0) * inv_rms.unsqueeze(-1)

    pre_raw = mixes[:, :M] * hc_scale[0] + hc_base[:M]
    post_raw = mixes[:, M : 2 * M] * hc_scale[1] + hc_base[M : 2 * M]
    comb_raw = mixes[:, 2 * M :].view(N, M, M) * hc_scale[2] + hc_base[2 * M :].view(
        M, M
    )

    post_mix = torch.sigmoid(post_raw) * hc_post_mult_value

    cm = comb_raw.softmax(-1) + hc_sinkhorn_eps
    cm = cm / (cm.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    comb_mix = cm.view(N, -1)

    pre_w = torch.sigmoid(pre_raw) + hc_pre_eps
    layer_input = (residual.float() * pre_w.unsqueeze(-1)).sum(1).bfloat16()
    return post_mix, comb_mix, layer_input


# Benchmark Utilities


def do_bench(fn, warmup=50, rep=300):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def compare_tensors(a, b):
    if torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan"), float("nan"), float("nan")
    d = (a.float() - b.float()).abs()
    return (
        d.max().item(),
        d.mean().item(),
        (d / b.float().abs().clamp(min=1e-8)).max().item(),
    )


def benchmark_config(
    num_tokens,
    hidden_size,
    hc_mult=4,
    n_splits=1,
    sinkhorn_repeat=10,
):
    M = hc_mult
    M2 = M * M
    M3 = 2 * M + M2
    H = hidden_size
    BH = _bh_for(H)
    device = "cuda"
    rms_eps = hc_pre_eps = hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 1.0
    common = (
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    torch.manual_seed(42)
    gm = (
        torch.randn(n_splits, num_tokens, M3, dtype=torch.float32, device=device) * 0.01
    )
    gs = (
        torch.randn(n_splits, num_tokens, dtype=torch.float32, device=device).abs()
        * 100
        + 1.0
    )
    sc = torch.randn(3, dtype=torch.float32, device=device) * 0.1
    ba = torch.randn(M3, dtype=torch.float32, device=device) * 0.1
    res = torch.randn(num_tokens, M, hidden_size, dtype=torch.bfloat16, device=device)

    # Reference
    pr, cr, lr = big_fuse_ref_pytorch(gm, gs, sc, ba, res, *common)

    # TileLang
    pt = torch.empty(num_tokens, M, dtype=torch.float32, device=device)
    ct = torch.empty(num_tokens, M2, dtype=torch.float32, device=device)
    lt = torch.empty(num_tokens, H, dtype=torch.bfloat16, device=device)

    def tl_call():
        mhc_pre_big_fuse_tilelang(gm, gs, sc, ba, res, pt, ct, lt, *common)

    tl_call()
    torch.cuda.synchronize()

    # Triton-TLX
    px = torch.empty(num_tokens, M, dtype=torch.float32, device=device)
    cx = torch.empty(num_tokens, M2, dtype=torch.float32, device=device)
    lx = torch.empty(num_tokens, H, dtype=torch.bfloat16, device=device)

    def tr_call():
        mhc_pre_big_fuse_triton(gm, gs, sc, ba, res, px, cx, lx, *common)

    tr_call()
    torch.cuda.synchronize()

    # ── Correctness ─────────────────────────────────────
    sep = "═" * 90
    print(f"\n{sep}")
    print(
        f"  N={num_tokens:<6d}  H={hidden_size:<5d}  M={M}  BH={BH}  NH={H//BH}"
        f"  n_splits={n_splits}  sinkhorn={sinkhorn_repeat}"
    )
    print(sep)

    print(
        f"\n  {'':14s}  {'── TileLang vs Ref ──':>36s}"
        f"   {'── Triton-TLX vs Ref ──':>36s}"
    )
    print(
        f"  {'Output':14s}  {'max_abs':>10s} {'mean_abs':>10s} {'max_rel':>10s}"
        f"   {'max_abs':>10s} {'mean_abs':>10s} {'max_rel':>10s}"
    )
    print(f"  {'─'*14}  {'─'*10} {'─'*10} {'─'*10}" f"   {'─'*10} {'─'*10} {'─'*10}")
    for name, ref, tl_o, tr_o in [
        ("post_mix", pr, pt, px),
        ("comb_mix", cr, ct, cx),
        ("layer_input", lr, lt, lx),
    ]:
        a1, b1, c1 = compare_tensors(tl_o, ref)
        a2, b2, c2 = compare_tensors(tr_o, ref)
        print(
            f"  {name:14s}  {a1:10.2e} {b1:10.2e} {c1:10.2e}"
            f"   {a2:10.2e} {b2:10.2e} {c2:10.2e}"
        )

    print(f"\n  TileLang vs Triton-TLX:")
    print(f"  {'Output':14s}  {'max_abs':>10s} {'mean_abs':>10s} {'max_rel':>10s}")
    print(f"  {'─'*14}  {'─'*10} {'─'*10} {'─'*10}")
    for name, tl_o, tr_o in [
        ("post_mix", pt, px),
        ("comb_mix", ct, cx),
        ("layer_input", lt, lx),
    ]:
        a, b, c = compare_tensors(tl_o, tr_o)
        print(f"  {name:14s}  {a:10.2e} {b:10.2e} {c:10.2e}")

    # ── Performance ─────────────────────────────────────
    ms_tl = do_bench(tl_call)
    ms_tr = do_bench(tr_call)

    bytes_rw = (
        n_splits * num_tokens * M3 * 4  # gemm_out_mul read
        + n_splits * num_tokens * 4  # gemm_out_sqrsum read
        + 3 * 4
        + M3 * 4  # hc_scale + hc_base read
        + num_tokens * M * H * 2  # residual read (bf16)
        + num_tokens * M * 4  # post_mix write
        + num_tokens * M2 * 4  # comb_mix write
        + num_tokens * H * 2  # layer_input write (bf16)
    )
    flops = num_tokens * (
        n_splits
        + 3  # sqrsum accumulate + rsqrt
        + M3 * (n_splits + 1)  # mix accumulate + scale
        + M * 8  # post_mix sigmoid
        + M2 * 2  # comb_mix scale+bias
        + M2 * 6  # softmax (exp+sum+div+eps)
        + M2 * 3  # col_norm (sum+div+eps)
        + (sinkhorn_repeat - 1) * M2 * 6  # sinkhorn iters
        + M * 8  # pre_mix sigmoid
        + M * H * 2  # weighted sum (mul + add)
    )
    bw_tl = bytes_rw / (ms_tl * 1e-3) / 1e9
    bw_tr = bytes_rw / (ms_tr * 1e-3) / 1e9
    gf_tl = flops / (ms_tl * 1e-3) / 1e9
    gf_tr = flops / (ms_tr * 1e-3) / 1e9
    speedup = ms_tl / ms_tr

    print(
        f"\n  Performance  (data={bytes_rw/1e6:.1f} MB,  FLOPs={flops/1e6:.1f} MFLOP)"
    )
    print(
        f"  {'':14s}  {'Latency':>10s}  {'BW (GB/s)':>10s}"
        f"  {'GFLOPS':>10s}  {'TFLOPS':>10s}"
    )
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(
        f"  {'TileLang':14s}  {ms_tl:8.3f}ms  {bw_tl:10.1f}"
        f"  {gf_tl:10.2f}  {gf_tl/1000:10.4f}"
    )
    print(
        f"  {'Triton-TLX':14s}  {ms_tr:8.3f}ms  {bw_tr:10.1f}"
        f"  {gf_tr:10.2f}  {gf_tr/1000:10.4f}"
    )

    if speedup > 1:
        tag = f"Triton-TLX faster by {speedup:.2f}x"
    else:
        tag = f"TileLang faster by {1/speedup:.2f}x"
    print(f"\n  Speedup (TL/TR) = {speedup:.3f}x  →  {tag}")

    return {
        "N": num_tokens,
        "H": hidden_size,
        "M": M,
        "ms_tl": ms_tl,
        "ms_tr": ms_tr,
        "bw_tl": bw_tl,
        "bw_tr": bw_tr,
        "gf_tl": gf_tl,
        "gf_tr": gf_tr,
        "speedup": speedup,
    }


def main():
    configs = [
        (512, 1280, 4),
        (512, 2560, 4),
        (512, 4096, 4),
        (1024, 1280, 4),
        (1024, 2560, 4),
        (1024, 4096, 4),
        (2048, 1280, 4),
        (2048, 2560, 4),
        (2048, 4096, 4),
        (8192, 1280, 4),
        (8192, 2560, 4),
        (8192, 4096, 4),
    ]
    gpu = torch.cuda.get_device_name()
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("═" * 90)
    print("  mhc_pre_big_fuse  —  TileLang vs Triton-TLX ")
    print(f"  GPU: {gpu}  ({mem:.1f} GB)")
    print("═" * 90)

    results = []
    for n, h, m in configs:
        results.append(benchmark_config(n, h, m))

    print(f"\n\n{'═' * 90}")
    print("  SUMMARY TABLE")
    print(f"{'═' * 90}")
    print(
        f"  {'N':>6s}  {'H':>5s}  {'BH':>4s}  NH │"
        f" {'TL ms':>8s}  {'TR ms':>8s} │"
        f" {'TL BW':>8s}  {'TR BW':>8s} │ {'Speed':>8s}"
    )
    print(
        f"  {'─'*6}  {'─'*5}  {'─'*4}  ── │"
        f" {'─'*8}  {'─'*8} │"
        f" {'─'*8}  {'─'*8} │ {'─'*8}"
    )
    for r in results:
        bh = _bh_for(r["H"])
        nh = r["H"] // bh
        if r["speedup"] > 1:
            tag = f"TR {r['speedup']:.2f}x"
        else:
            tag = f"TL {1/r['speedup']:.2f}x"
        print(
            f"  {r['N']:6d}  {r['H']:5d}  {bh:4d}  {nh:2d} │"
            f" {r['ms_tl']:8.3f}  {r['ms_tr']:8.3f} │"
            f" {r['bw_tl']:7.1f}G  {r['bw_tr']:7.1f}G │ {tag:>8s}"
        )
    print(f"\n  TR = Triton-TLX faster,  TL = TileLang faster")
    print("═" * 90)


if __name__ == "__main__":
    main()
