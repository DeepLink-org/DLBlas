import triton
import triton.language as tl
import torch
from dlblas.utils.device_utils import NUM_CORES, DEVICE


@triton.jit
def _mhc_post_kernel(
    comb_res_mix_ptr,
    residual_ptr,
    post_layer_mix_ptr,
    x_ptr,
    out_ptr,
    stride_comb_n,
    stride_comb_hc1,
    stride_comb_hc2,
    stride_residual_n,
    stride_residual_hc,
    stride_residual_h,
    stride_post_layer_mix_n,
    stride_post_layer_mix_hc,
    stride_xn,
    stride_xh,
    stride_out_n,
    stride_out_hc,
    stride_out_h,
    N,
    H: tl.constexpr,
    HC: tl.constexpr,
    H_BLK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    core_idx = tl.program_id(0)
    offs_hc1 = tl.arange(0, HC)[:, None]
    offs_j = tl.arange(0, HC)[None, :]
    offs_hblk = tl.arange(0, H_BLK)[None, :]

    for pid_n in range(core_idx, N, NUM_CORES):
        a_ptrs = (
            comb_res_mix_ptr
            + pid_n * stride_comb_n
            + offs_hc1 * stride_comb_hc1
            + offs_j * stride_comb_hc2
        )
        a_data = tl.trans(tl.load(a_ptrs))
        post_layer_mix_ptrs = (
            post_layer_mix_ptr
            + pid_n * stride_post_layer_mix_n
            + tl.arange(0, HC) * stride_post_layer_mix_hc
        )
        post_layer_mix_data = tl.load(post_layer_mix_ptrs)
        n_blocks = tl.cdiv(H, H_BLK)
        for blk_idx in range(n_blocks):
            start_h = blk_idx * H_BLK
            residual_offs_h = start_h + offs_hblk
            residual_ptrs = (
                residual_ptr
                + pid_n * stride_residual_n
                + offs_hc1 * stride_residual_hc
                + residual_offs_h * stride_residual_h
            )
            residual_data = tl.load(residual_ptrs, mask=residual_offs_h < H)
            dot_res = tl.dot(a_data, residual_data.to(tl.float32))
            x_offs_h = start_h + tl.arange(0, H_BLK)
            x_ptrs = x_ptr + pid_n * stride_xn + x_offs_h * stride_xh
            x_data = tl.load(x_ptrs, mask=x_offs_h < H, other=0.0)
            result = (post_layer_mix_data[:, None] * x_data[None, :]) + dot_res
            out_ptrs = (
                out_ptr
                + pid_n * stride_out_n
                + offs_hc1 * stride_out_hc
                + (start_h + offs_hblk) * stride_out_h
            )
            tl.store(out_ptrs, result.to(tl.bfloat16), mask=offs_hblk < (H - start_h))


def mhc_post_triton(
    comb_res_mix: torch.Tensor,  # (N, HC, HC) float32
    residual: torch.Tensor,  # (N, HC, H) bfloat16
    post_layer_mix: torch.Tensor,  # (N, HC) float32
    x: torch.Tensor,  # (N, H) bfloat16
    out: torch.Tensor,  # (N, HC, H) bfloat16 (output)
    hc: int,
    hidden: int,
    h_blk: int = 512,
):
    # Validate shapes and dtypes
    assert (
        comb_res_mix.shape == (out.shape[0], hc, hc)
        and comb_res_mix.dtype == torch.float32
    )
    assert (
        residual.shape == (out.shape[0], hc, hidden)
        and residual.dtype == torch.bfloat16
    )
    assert (
        post_layer_mix.shape == (out.shape[0], hc)
        and post_layer_mix.dtype == torch.float32
    )
    assert x.shape == (out.shape[0], hidden) and x.dtype == torch.bfloat16
    assert out.shape == (out.shape[0], hc, hidden) and out.dtype == torch.bfloat16

    _mhc_post_kernel[(NUM_CORES,)](
        comb_res_mix,
        residual,
        post_layer_mix,
        x,
        out,
        comb_res_mix.stride(0),
        comb_res_mix.stride(1),
        comb_res_mix.stride(2),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post_layer_mix.stride(0),
        post_layer_mix.stride(1),
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=comb_res_mix.shape[0],
        H=hidden,
        HC=hc,
        H_BLK=h_blk,
        NUM_CORES=NUM_CORES,
    )
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    mhc_post_triton(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def generate_test_data(
    n: int,
    h: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        (n, hc_mult, hc_mult), dtype=torch.float32, device=device
    )

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def test(n: int, h: int) -> None:
    print(f"Testing mhc_post with {n=} {h=}")
    test_data = generate_test_data(n=n, h=h, hc_mult=16, device=DEVICE)
    out_tl = mhc_post(**test_data)
    out_ref = mhc_post_ref(**test_data)
    torch.testing.assert_close(out_tl, out_ref)
    print(f"mhc_post with {n=} {h=} success")


def benchmark():
    n = 4096
    for h in [1280, 2560, 7168]:
        test_data = generate_test_data(n=n, h=h, hc_mult=16, device=DEVICE)
        out_tl = mhc_post(**test_data)
        out_ref = mhc_post_ref(**test_data)
        torch.testing.assert_close(out_tl, out_ref)
        configs = []
        configs.append(
            triton.testing.Benchmark(
                x_names=["cnt"],  # Argument names to use as an x-axis for the plot
                # x_vals=[128 * i for i in range(10, 15)],  # Different possible values for `x_name`
                x_vals=[1],  # NOTE: the tunning framework specialized to one shape
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                line_vals=["triton", "torch"],  # Label name for the lines
                line_names=["Triton", "Torch"],  # Line styles
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",  # Label name for the y-axis
                plot_name="mhc-post-performance-" + f"bf16-[n={n} h={h}]",
                args={},
            )
        )

        @triton.testing.perf_report(configs)
        def benchmark(cnt, provider):
            warmup = 500
            rep = 500
            quantiles = [0.5, 0.2, 0.8]
            if provider == "torch":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: mhc_post_ref(**test_data),
                    quantiles=quantiles,
                    warmup=warmup,
                    rep=rep,
                )
            if provider == "triton":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: mhc_post(**test_data),
                    quantiles=quantiles,
                    warmup=warmup,
                    rep=rep,
                )
            return ms, max_ms, min_ms

        benchmark.run(show_plots=False, print_data=True)
        print("run benchmark success")


def main():
    for n in [4096]:
        for h in [1280, 2560, 7168]:
            test(n=n, h=h)


if __name__ == "__main__":
    benchmark()
