import torch
import triton
import triton.language as tl
from dlblas.utils.device_utils import NUM_CORES, DEVICE


def _bh_for(H: int) -> int:
    for b in (1024, 512, 256, 128, 64):
        if H % b == 0:
            return b
    return 32


@triton.jit
def _k_fuse_tlx(
    gm_ptr,  # [NS, N, M3]  f32   gemm_out_mul
    gs_ptr,  # [NS, N]      f32   gemm_out_sqrsum
    sc_ptr,  # [3]           f32   hc_scale
    ba_ptr,  # [M3]          f32   hc_base
    res_ptr,  # [N, M, H]    bf16  residual
    pp_ptr,  # [N, M]        f32   post_mix  (out)
    cp_ptr,  # [N, M*M]      f32   comb_mix  (out)
    lp_ptr,  # [N, H]        bf16  layer_input (out)
    N,
    H,
    eps_rms: tl.constexpr,
    eps_pre: tl.constexpr,
    eps_sk: tl.constexpr,
    pmul: tl.constexpr,
    M: tl.constexpr,
    M3: tl.constexpr,
    BH: tl.constexpr,
    SR: tl.constexpr,
    NS: tl.constexpr,
    NH: tl.constexpr,
):
    """
    Fused 1D kernel: one block per token.
    Steps: RMS → mix_accum → post_mix → sinkhorn → pre_mix → weighted_sum
    The weighted_sum loop uses tlx.async_load for 2-stage software pipelining.
    """
    pid = tl.program_id(0)
    mo = tl.arange(0, M)  # [M]
    ri = mo[:, None]  # [M, 1]
    ci = mo[None, :]  # [1, M]
    ho = tl.arange(0, BH)  # [BH]
    sq = 0.0
    for s in range(NS):
        sq = sq + tl.load(gs_ptr + s * N + pid)
    inv = tl.math.rsqrt(sq / M / H + eps_rms)
    vp = tl.zeros([M], dtype=tl.float32)
    vpo = tl.zeros([M], dtype=tl.float32)
    vc = tl.zeros([M, M], dtype=tl.float32)
    for s in range(NS):
        b = s * N * M3 + pid * M3
        vp = vp + tl.load(gm_ptr + b + mo)
        vpo = vpo + tl.load(gm_ptr + b + M + mo)
        vc = vc + tl.load(gm_ptr + b + 2 * M + ri * M + ci)
    vp = vp * inv
    vpo = vpo * inv
    vc = vc * inv
    s1 = tl.load(sc_ptr + 1)
    pob = tl.load(ba_ptr + M + mo)
    tl.store(pp_ptr + pid * M + mo, tl.sigmoid(vpo * s1 + pob) * pmul)
    s2 = tl.load(sc_ptr + 2)
    cb = tl.load(ba_ptr + 2 * M + ri * M + ci)
    cm = vc * s2 + cb
    mx = tl.max(cm, axis=1)
    cm = tl.exp(cm - mx[:, None])
    rs = tl.sum(cm, axis=1)
    cm = cm / rs[:, None] + eps_sk
    cs = tl.sum(cm, axis=0)
    cm = cm / (cs[None, :] + eps_sk)
    for _ in range(SR - 1):
        rs = tl.sum(cm, axis=1)
        cm = cm / (rs[:, None] + eps_sk)
        cs = tl.sum(cm, axis=0)
        cm = cm / (cs[None, :] + eps_sk)
    tl.store(cp_ptr + pid * M * M + ri * M + ci, cm)
    s0 = tl.load(sc_ptr + 0)
    pb = tl.load(ba_ptr + mo)
    pw = tl.sigmoid(vp * s0 + pb) + eps_pre  # [M]
    rbase = res_ptr + pid * M * H  # Base pointer for this token
    # Build 2D offset pattern: [M, BH]
    # offset[m, bh] = m*H + ih*BH + bh
    off_m = mo[:, None] * H  # [M, 1]
    off_h = ho[None, :]  # [1, BH]
    for ih in range(NH):
        # Load tile: [M, BH] bf16
        tile_ptr = rbase + off_m + ih * BH + off_h
        tile = tl.load(tile_ptr).to(tl.float32)  # [M, BH]
        acc = tl.sum(pw[:, None] * tile, axis=0)  # [BH] f32
        # Store output slice
        tl.store(
            lp_ptr + pid * H + ih * BH + ho,
            acc.to(tl.bfloat16),
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
    assert NH >= 2, f"Need NH≥2 for 2-stage pipeline (H={H}, BH={BH}, NH={NH})"

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
        H,
        eps_rms=rms_eps,
        eps_pre=hc_pre_eps,
        eps_sk=hc_sinkhorn_eps,
        pmul=hc_post_mult_value,
        M=M,
        M3=M3,
        BH=BH,
        SR=sinkhorn_repeat,
        NS=n_splits,
        NH=NH,
    )


@triton.jit
def _mhc_pre_gemm_sqrsum_kernel(
    x_ptr,  # (num_tokens, hc_hidden_size), bfloat16
    fn_ptr,  # (hc_mult3, hc_hidden_size), float32
    out_ptr,  # (num_tokens, hc_mult3), float32
    sqrsum_ptr,  # (num_tokens,), float32
    num_tokens,
    hc_mult3,  # <= 32
    hc_hidden_size,
    token_block: tl.constexpr = 32,
    hidden_block: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    offs_m = pid * token_block + tl.arange(0, token_block)
    mask_m = offs_m < num_tokens
    acc_sqr = tl.zeros((token_block,), dtype=tl.float32)
    acc_out = tl.zeros((token_block, 32), dtype=tl.float32)
    for k in range(0, hc_hidden_size, hidden_block):
        offs_k = k + tl.arange(0, hidden_block)
        x_ptrs = x_ptr + offs_m[:, None] * hc_hidden_size + offs_k[None, :]
        x_block_bf16 = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        x_block_f32 = x_block_bf16.to(tl.float32)
        block_sqr = tl.sum(x_block_f32 * x_block_f32, axis=1)
        acc_sqr += block_sqr
        offs_n = tl.arange(0, 32)
        mask_fn_row = offs_n < hc_mult3
        fn_ptrs = fn_ptr + offs_n[:, None] * hc_hidden_size + offs_k[None, :]
        fn_block = tl.load(fn_ptrs, mask=mask_fn_row[:, None], other=0.0)
        fn_block_T = tl.trans(fn_block)
        block_out = tl.dot(x_block_f32, fn_block_T, allow_tf32=True)
        acc_out += block_out

    sqrsum_ptrs = sqrsum_ptr + offs_m
    tl.store(sqrsum_ptrs, acc_sqr, mask=mask_m)
    offs_n_out = tl.arange(0, 32)
    out_ptrs = out_ptr + offs_m[:, None] * hc_mult3 + offs_n_out[None, :]
    mask_out = mask_m[:, None] & (offs_n_out[None, :] < hc_mult3)
    tl.store(out_ptrs, acc_out, mask=mask_out)


def mhc_pre_gemm_sqrsum_triton(x, fn, out, sqrsum, hc_mult3, hc_hidden_size):
    token_block = 32
    num_tokens = x.shape[0]
    grid = (triton.cdiv(num_tokens, token_block),)
    _mhc_pre_gemm_sqrsum_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        num_tokens,
        hc_mult3,
        hc_hidden_size,
        token_block=32,
        hidden_block=256,
    )


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor; TileLang version of mhc_pre_gemm_sqrsum doesn't support this

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    assert (
        n_splits == 1
    ), "The simple TileLang version gemm_sqrsum doesn't support split-k"
    mhc_pre_gemm_sqrsum_triton(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn_flat,
        gemm_out_mul.squeeze(0),
        gemm_out_sqrsum.squeeze(0),
        hc_mult3,
        hc_mult * hidden_size,
    )
    mhc_pre_big_fuse_triton(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
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
    )
    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )
    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input


def generate_test_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
) -> dict[str, torch.Tensor | float]:
    """Generate test data for big fuse operator."""
    torch.random.manual_seed(42)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = DEVICE
    residual = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1
    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def test(n: int, hidden_size: int, hc_mult: int) -> None:
    print(f"Testing mhc_pre with {n=} {hidden_size=} {hc_mult=}")
    test_data = generate_test_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )
    # Forward pass with big fuse
    post_mix_fused, comb_mix_fused, layer_input_fused = mhc_pre(**test_data)
    # Forward pass with reference
    post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre_ref(**test_data)
    # Compare outputs
    torch.testing.assert_close(post_mix_fused, post_mix_ref)
    torch.testing.assert_close(comb_mix_fused, comb_mix_ref)
    torch.testing.assert_close(layer_input_fused, layer_input_ref)
    print(f"Testing mhc_pre with {n=} {hidden_size=} {hc_mult=} success")


def main():
    for n1 in [512, 1024, 2048, 8192]:
        for hidden_size in [1280, 2560, 4096]:
            for hc_mult in [4]:
                test(n=n1, hidden_size=hidden_size, hc_mult=hc_mult)


if __name__ == "__main__":
    main()
