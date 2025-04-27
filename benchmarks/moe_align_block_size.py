import argparse
import itertools

import torch
import triton
import triton.language as tl
from dlblas.kernels.moe_align_block_size import moe_align_block_size_triton
from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
from vllm import _custom_ops as ops

USE_RANDOM_PERM = False




def calculate_diff(num_tokens, num_experts=256, block_size=128, topk=8):
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(num_tokens)
        ]
    )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids_cuda = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    token_cnts_buffer = torch.zeros(
        (num_experts + 1) * num_experts, dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    sorted_ids_vllm = torch.empty_like(sorted_ids_cuda)
    sorted_ids_vllm.fill_(topk_ids.numel())
    expert_ids_vllm = torch.zeros_like(expert_ids_cuda)
    num_tokens_post_pad_vllm = torch.empty_like(num_tokens_post_pad_cuda)

    # compare the performance of cuda, triton and vllm implementation
    sgl_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        token_cnts_buffer,
        cumsum_buffer,
    )
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_triton,
        expert_ids_triton,
        num_tokens_post_pad_triton,
    )

    try:
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids_vllm,
            expert_ids_vllm,
            num_tokens_post_pad_vllm,
        )
        print(f"✅ VLLM implementation works with {num_experts} experts!")
        vllm_works = True
    except Exception as e:
        print(f"❌ VLLM implementation failed with {num_experts} experts: {e}")
        vllm_works = False

    if torch.allclose(expert_ids_cuda, expert_ids_triton) and torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton
    ):
        print("✅ SGL and Triton implementations match")
    else:
        print("❌ SGL and Triton implementations do not match")
        print("SGL expert_ids:", expert_ids_cuda)
        print("Triton expert_ids:", expert_ids_triton)
        print("SGL num_tokens_post_pad:", num_tokens_post_pad_cuda)
        print("Triton num_tokens_post_pad:", num_tokens_post_pad_triton)

    if (
        vllm_works
        and torch.allclose(expert_ids_cuda, expert_ids_vllm)
        and torch.allclose(num_tokens_post_pad_cuda, num_tokens_post_pad_vllm)
    ):
        print("✅ SGL and VLLM implementations match")
    else:
        if not vllm_works:
            print("⚠️ VLLM comparison skipped due to failure")
        else:
            print("❌ SGL and VLLM implementations do not match")
            print("SGL expert_ids:", expert_ids_cuda)
            print("VLLM expert_ids:", expert_ids_vllm)
            print("SGL num_tokens_post_pad:", num_tokens_post_pad_cuda)
            print("VLLM num_tokens_post_pad:", num_tokens_post_pad_vllm)


# Test range
num_tokens_range = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_experts_range = [8, 32, 64, 128, 256]
topk_range = [1, 2, 4, 8]

configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    for i in range(num_tokens):
        topk_ids[i, :] = torch.randperm(num_experts, dtype=torch.int32, device="cuda")[
            :topk
        ]
    return topk_ids


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl", "triton", "vllm"],
        line_names=["SGL", "Triton", "VLLM"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    block_size = 128

    if USE_RANDOM_PERM:
        topk_ids = get_topk_ids(num_tokens, num_experts, topk)
    else:
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, topk),
            dtype=torch.int32,
            device="cuda",
        )

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "sgl":

        def sgl_moe_align_block_size_with_empty(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        ):
            token_cnts_buffer = torch.empty(
                (num_experts + 1) * num_experts,
                dtype=torch.int32,
                device=topk_ids.device,
            )
            cumsum_buffer = torch.empty(
                num_experts + 1, dtype=torch.int32, device=topk_ids.device
            )

            sgl_moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
                token_cnts_buffer,
                cumsum_buffer,
            )

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sgl_moe_align_block_size_with_empty(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size_triton(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
            ),
            quantiles=quantiles,
        )
    else:  # vllm
        try:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: ops.moe_align_block_size(
                    topk_ids,
                    num_experts,
                    block_size,
                    sorted_ids.clone(),
                    expert_ids.clone(),
                    num_tokens_post_pad.clone(),
                ),
                quantiles=quantiles,
            )
        except RuntimeError as e:
            print(f"❌ VLLM benchmark failed with {num_experts} experts: {e}")
            # Return extreme values to indicate failure in the chart
            return float("inf"), float("inf"), float("inf")

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/moe_align_blocks/",
        help="Path to save moe align benchmark results",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=256,
        choices=[8, 16, 32, 64, 128, 256],
        help="Number of experts for benchmark",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        choices=[2, 4, 8],
        help="Top-k value for benchmark",
    )
    parser.add_argument(
        "--skip_full_benchmark",
        action="store_true",
        help="Only run the calculate_diff function, skip full benchmarking",
    )
    args = parser.parse_args()

    calculate_diff(num_tokens=1024, num_experts=args.num_experts, topk=args.topk)

    if not args.skip_full_benchmark:
        print(f"\n📊 Running performance benchmark for {args.num_experts} experts...")
        benchmark.run(print_data=True)
