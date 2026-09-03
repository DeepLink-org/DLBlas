"""Run one real multi-rank Torch Planning correctness/performance round."""

from __future__ import annotations

import argparse
import importlib
import os
import time


def parse_args() -> argparse.Namespace:
    """Parse one candidate's module, case, and timing controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        default="dlblas.kernels.ascend.moonep.planning_torch_ascend",
    )
    parser.add_argument("--case", default="balanced_epn16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--num-sms",
        type=int,
        default=None,
        help="override ctx['num_sms'] for implementations that expose it",
    )
    parser.add_argument(
        "--dedup-mode", choices=("zero", "current", "src"), default="current"
    )
    return parser.parse_args()


def main() -> None:
    """Compile, validate, and benchmark one candidate on both physical ranks."""
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    cache_root = os.environ["TRITON_CACHE_DIR"]
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, f"rank-{local_rank}")

    # Triton may snapshot its cache directory during torch-npu import, so all
    # accelerator imports intentionally happen after the rank-local override.
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from dlblas.kernels.ascend.moonep.planning_test_utils_ascend import (
        PLANNING_CASES,
        assert_planning_outputs_equal_all_ranks,
        init_case,
        launch_planning_torch_reference,
        make_topk,
        npu_index_for_local_rank,
        planning_invariant_errors,
    )

    device_ids = [int(value) for value in os.environ["MOONEP_NPU_IDS"].split(",")]
    torch.npu.set_device(npu_index_for_local_rank(local_rank, device_ids))
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()

    case = next(item for item in PLANNING_CASES if item.name == args.case)
    ctx = init_case(case, rank, R)
    if args.num_sms is not None:
        if args.num_sms <= 0:
            raise ValueError("--num-sms must be positive")
        ctx["num_sms"] = args.num_sms
    topk, tokens_per_expert = make_topk(case, rank, R)
    implementation = importlib.import_module(args.module)

    cu_seqlens, plan, src = implementation.launch_planning(
        ctx, topk, tokens_per_expert, dedup_mode=args.dedup_mode
    )
    ref_cu_seqlens, ref_plan, ref_src = launch_planning_torch_reference(
        ctx, topk, tokens_per_expert, dedup_mode=args.dedup_mode
    )
    assert_planning_outputs_equal_all_ranks(
        cu_seqlens,
        plan,
        src,
        ref_cu_seqlens,
        ref_plan,
        ref_src,
        rank,
        R,
        dedup_mode=args.dedup_mode,
        label_prefix=f"{implementation.IMPLEMENTATION_NAME}/",
    )
    errors = planning_invariant_errors(
        case,
        ctx,
        topk,
        tokens_per_expert,
        plan,
        cu_seqlens,
        src,
        dedup_mode=args.dedup_mode,
    )
    if errors:
        raise AssertionError("; ".join(errors[:5]))

    for _ in range(args.warmup):
        implementation.launch_planning(
            ctx, topk, tokens_per_expert, dedup_mode=args.dedup_mode
        )
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeat):
        dist.barrier()
        started = time.perf_counter()
        implementation.launch_planning(
            ctx, topk, tokens_per_expert, dedup_mode=args.dedup_mode
        )
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    peak = int(torch.npu.max_memory_allocated())
    print(
        f"ROUND_RESULT rank={rank} case={case.name} "
        f"dedup_mode={args.dedup_mode} correctness=pass "
        f"launches={ctx['_planning_launch_count']} median_ms={samples[len(samples)//2]:.6f} "
        f"min_ms={samples[0]:.6f} peak_bytes={peak}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
