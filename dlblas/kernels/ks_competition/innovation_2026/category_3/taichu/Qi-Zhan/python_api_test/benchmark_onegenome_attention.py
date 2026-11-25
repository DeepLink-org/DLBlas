#!/usr/bin/env python3
"""OneGenomeRice correctness gate and tecoops attention microbenchmark.

This test models one tensor-parallel rank of OneGenomeRice:
  4 query heads, 2 KV heads, head size 64, paged-cache block size 32.

It tests reshape_and_cache and flash_attn_varlen_func as one pipeline. The
reference computes only selected query rows, so long-context T1-T4 cases do
not allocate a quadratic attention matrix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tecoops


NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 32
DTYPE = torch.float16
DEVICE = "sdaa"
DEFAULT_ATOL = 2e-2

OFFICIAL_CONTEXTS = {
    # The OneGenome tokenizer emits two tokens per DNA base minus one.
    "T1": 32_767,
    "T2": 65_535,
    "T3": 131_071,
    "T4": 204_799,
}


@dataclass(frozen=True)
class Case:
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


@dataclass
class Inputs:
    case: Case
    q_cpu: torch.Tensor
    key_cpu: torch.Tensor
    value_cpu: torch.Tensor
    q: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    slot_mapping: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    block_table: torch.Tensor
    out: torch.Tensor
    key_offsets: list[int]
    q_offsets: list[int]


QUICK_CASES = (
    Case("rice_prefill_924_b1", (924,), (924,)),
    Case("rice_prefill_924_b8", (924,) * 8, (924,) * 8),
    Case("rice_chunked_prefill_b1", (512,), (1024,)),
    Case("rice_decode_925_b1", (1,), (925,)),
    Case("rice_decode_925_b128", (1,) * 128, tuple(925 + i % 100 for i in range(128))),
    Case("rice_mixed", (1, 16, 32, 64), (1024, 128, 256, 1024)),
)

# These cases do not compare against a different Attention implementation.
# They verify that batching is only a scheduler-level loop over the exact B=1
# model kernel. The long case crosses the decode window boundary so a hidden
# length- or batch-dependent implementation switch cannot pass unnoticed.
REPLAY_CASES = (
    Case("same_kernel_prefill_b8", (64,) * 8, (64,) * 8),
    Case(
        "same_kernel_decode_b128",
        (1,) * 128,
        tuple(925 + index % 100 for index in range(128)),
    ),
    Case("same_kernel_mixed_b4", (1, 16, 32, 64), (1024, 128, 256, 1024)),
    Case("same_kernel_long_decode_b2", (1, 1), (32_767, 32_768)),
)

FUSED_REPLAY_CONTEXTS = (925, 4097)


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def prefix_sum(values: tuple[int, ...]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def make_inputs(case: Case, seed: int) -> Inputs:
    if len(case.q_lens) != len(case.kv_lens):
        raise ValueError(f"{case.name}: q_lens and kv_lens must have equal length")
    if any(q_len > kv_len for q_len, kv_len in zip(case.q_lens, case.kv_lens)):
        raise ValueError(f"{case.name}: each q_len must be <= kv_len")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    total_q = sum(case.q_lens)
    total_kv = sum(case.kv_lens)
    q_width = NUM_Q_HEADS * HEAD_SIZE
    kv_width = NUM_KV_HEADS * HEAD_SIZE

    # The registered Mixtral implementation passes compact Q, K, and V
    # tensors. Keep the observed token strides (256 for Q, 128 for K/V) so
    # decode fast-path gates are exercised exactly as they are in vLLM.
    q_cpu = torch.randn(
        total_q, q_width, dtype=DTYPE, generator=generator
    ).view(total_q, NUM_Q_HEADS, HEAD_SIZE)
    key_cpu = torch.randn(
        total_kv, kv_width, dtype=DTYPE, generator=generator
    ).view(total_kv, NUM_KV_HEADS, HEAD_SIZE)
    value_cpu = torch.randn(
        total_kv, kv_width, dtype=DTYPE, generator=generator
    ).view(total_kv, NUM_KV_HEADS, HEAD_SIZE)

    blocks_per_seq = [cdiv(kv_len, BLOCK_SIZE) for kv_len in case.kv_lens]
    total_blocks = sum(blocks_per_seq)
    physical_blocks = torch.randperm(total_blocks, generator=generator).tolist()

    block_table_cpu = torch.zeros(
        len(case.q_lens), max(blocks_per_seq), dtype=torch.int32
    )
    slot_mapping_cpu = torch.empty(total_kv, dtype=torch.int64)
    key_offsets = prefix_sum(case.kv_lens)

    next_block = 0
    for batch_idx, (kv_len, block_count) in enumerate(
        zip(case.kv_lens, blocks_per_seq)
    ):
        blocks = physical_blocks[next_block : next_block + block_count]
        next_block += block_count
        block_table_cpu[batch_idx, :block_count] = torch.tensor(
            blocks, dtype=torch.int32
        )
        token_offset = key_offsets[batch_idx]
        for token_idx in range(kv_len):
            logical_block = token_idx // BLOCK_SIZE
            block_offset = token_idx % BLOCK_SIZE
            slot_mapping_cpu[token_offset + token_idx] = (
                blocks[logical_block] * BLOCK_SIZE + block_offset
            )

    q = q_cpu.to(DEVICE)
    key = key_cpu.to(DEVICE)
    value = value_cpu.to(DEVICE)
    key_cache = torch.zeros(
        total_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_SIZE,
        dtype=DTYPE,
        device=DEVICE,
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = slot_mapping_cpu.to(DEVICE)
    cu_seqlens_q = torch.tensor(prefix_sum(case.q_lens), dtype=torch.int32).to(
        DEVICE
    )
    seqused_k = torch.tensor(case.kv_lens, dtype=torch.int32).to(DEVICE)
    block_table = block_table_cpu.to(DEVICE)
    out = torch.empty_like(q)

    return Inputs(
        case=case,
        q_cpu=q_cpu,
        key_cpu=key_cpu,
        value_cpu=value_cpu,
        q=q,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        out=out,
        key_offsets=key_offsets,
        q_offsets=prefix_sum(case.q_lens),
    )


def run_reshape(inputs: Inputs) -> None:
    tecoops.reshape_and_cache(
        inputs.key,
        inputs.value,
        inputs.slot_mapping,
        inputs.key_cache,
        inputs.value_cache,
    )


def run_attention(inputs: Inputs) -> None:
    empty_int = torch.empty(0, dtype=torch.int32, device=DEVICE)
    tecoops.flash_attn_varlen_func(
        inputs.q,
        inputs.key_cache,
        inputs.value_cache,
        max(inputs.case.q_lens),
        inputs.cu_seqlens_q,
        max(inputs.case.kv_lens),
        empty_int,
        inputs.seqused_k,
        1.0 / math.sqrt(HEAD_SIZE),
        True,
        empty_int,
        inputs.block_table,
        False,
        inputs.out,
    )


def selected_rows(q_len: int, count: int) -> list[int]:
    candidates = [0, q_len - 1]
    if q_len > 2:
        candidates.extend([q_len // 4, q_len // 2, (3 * q_len) // 4])
    if count > len(candidates):
        step = max(1, q_len // count)
        candidates.extend(range(0, q_len, step))
    return sorted(set(max(0, min(q_len - 1, row)) for row in candidates))[:count]


def reference_row(
    inputs: Inputs, batch_idx: int, local_q_row: int
) -> torch.Tensor:
    q_start = inputs.q_offsets[batch_idx]
    kv_start = inputs.key_offsets[batch_idx]
    q_len = inputs.case.q_lens[batch_idx]
    kv_len = inputs.case.kv_lens[batch_idx]
    last_visible_key = kv_len - q_len + local_q_row

    q_row = inputs.q_cpu[q_start + local_q_row].float()
    key = inputs.key_cpu[kv_start : kv_start + last_visible_key + 1].float()
    value = inputs.value_cpu[kv_start : kv_start + last_visible_key + 1].float()
    result = torch.empty(NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float32)
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    scale = 1.0 / math.sqrt(HEAD_SIZE)

    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // group_size
        scores = torch.mv(key[:, kv_head, :], q_row[q_head]) * scale
        probabilities = torch.softmax(scores, dim=0)
        result[q_head] = torch.mv(value[:, kv_head, :].transpose(0, 1), probabilities)
    return result


def check_cache_samples(inputs: Inputs, sample_count: int = 32) -> dict:
    total_tokens = inputs.key_cpu.size(0)
    if total_tokens <= sample_count:
        token_indices = list(range(total_tokens))
    else:
        token_indices = torch.linspace(
            0, total_tokens - 1, steps=sample_count, dtype=torch.int64
        ).tolist()

    slots = inputs.slot_mapping.cpu()[token_indices]
    block_ids = torch.div(slots, BLOCK_SIZE, rounding_mode="floor")
    block_offsets = slots.remainder(BLOCK_SIZE)
    key_actual = inputs.key_cache[block_ids, :, block_offsets, :].cpu()
    value_actual = inputs.value_cache[block_ids, :, block_offsets, :].cpu()
    key_expected = inputs.key_cpu[token_indices]
    value_expected = inputs.value_cpu[token_indices]

    return {
        "samples": len(token_indices),
        "key_exact": bool(torch.equal(key_actual, key_expected)),
        "value_exact": bool(torch.equal(value_actual, value_expected)),
    }


def run_correctness_case(
    case: Case, seed: int, rows_per_sequence: int, atol: float
) -> dict:
    inputs = make_inputs(case, seed)
    run_reshape(inputs)
    run_attention(inputs)
    torch.sdaa.synchronize()

    cache_result = check_cache_samples(inputs)
    out_cpu = inputs.out.cpu().float()
    max_error = 0.0
    mean_errors: list[float] = []
    checked_rows = 0

    for batch_idx, q_len in enumerate(case.q_lens):
        for local_q_row in selected_rows(q_len, rows_per_sequence):
            expected = reference_row(inputs, batch_idx, local_q_row)
            actual = out_cpu[inputs.q_offsets[batch_idx] + local_q_row]
            error = (actual - expected).abs()
            max_error = max(max_error, float(error.max()))
            mean_errors.append(float(error.mean()))
            checked_rows += 1

    passed = (
        cache_result["key_exact"]
        and cache_result["value_exact"]
        and max_error <= atol
    )
    result = {
        "name": case.name,
        "batch_size": len(case.q_lens),
        "total_q_tokens": sum(case.q_lens),
        "max_kv_len": max(case.kv_lens),
        "checked_query_rows": checked_rows,
        "input_strides": {
            "query": list(inputs.q.stride()),
            "key": list(inputs.key.stride()),
            "value": list(inputs.value.stride()),
        },
        "inputs_contiguous": {
            "query": inputs.q.is_contiguous(),
            "key": inputs.key.is_contiguous(),
            "value": inputs.value.is_contiguous(),
        },
        "max_abs_error": max_error,
        "mean_abs_error": statistics.fmean(mean_errors),
        "atol": atol,
        "cache": cache_result,
        "passed": passed,
    }
    del inputs
    release_device_cache()
    return result


def run_batched_replay_equivalence_case(case: Case, seed: int) -> dict:
    """Compare one batched call with explicit calls to the same B=1 entry."""
    inputs = make_inputs(case, seed)
    run_reshape(inputs)
    run_attention(inputs)
    torch.sdaa.synchronize()
    batched_out = inputs.out.clone()

    replay_out = torch.empty_like(inputs.out)
    empty_int = torch.empty(0, dtype=torch.int32, device=DEVICE)
    for batch_idx, (q_len, kv_len) in enumerate(
        zip(case.q_lens, case.kv_lens)
    ):
        q_start = inputs.q_offsets[batch_idx]
        q_end = inputs.q_offsets[batch_idx + 1]
        cu_single = torch.tensor(
            (0, q_len), dtype=torch.int32, device=DEVICE
        )
        tecoops.flash_attn_varlen_func(
            inputs.q[q_start:q_end],
            inputs.key_cache,
            inputs.value_cache,
            q_len,
            cu_single,
            kv_len,
            empty_int,
            inputs.seqused_k[batch_idx : batch_idx + 1],
            1.0 / math.sqrt(HEAD_SIZE),
            True,
            empty_int,
            inputs.block_table[batch_idx : batch_idx + 1],
            False,
            replay_out[q_start:q_end],
        )

    torch.sdaa.synchronize()
    difference = (batched_out.cpu().float() - replay_out.cpu().float()).abs()
    exact = bool(torch.equal(batched_out.cpu(), replay_out.cpu()))
    result = {
        "name": case.name,
        "kind": "batched_vs_explicit_b1",
        "batch_size": len(case.q_lens),
        "max_kv_len": max(case.kv_lens),
        "bitwise_equal": exact,
        "max_abs_difference": float(difference.max()),
        "passed": exact,
    }
    del inputs, batched_out, replay_out, difference
    release_device_cache()
    return result


def run_fused_decode_equivalence_case(context_len: int, seed: int) -> dict:
    """Prove the performance fusion still launches the same decode math."""
    case = Case(f"same_kernel_fused_decode_{context_len}", (1,), (context_len,))

    def run_once(fused: bool) -> torch.Tensor:
        inputs = make_inputs(case, seed)
        # Match the model's reserved cache capacity so the fused append+decode
        # route is selected. Both routes must enter the same B=1 core for the
        # tested length; only append scheduling may differ.
        inputs.key_cache = torch.zeros(
            8192,
            NUM_KV_HEADS,
            BLOCK_SIZE,
            HEAD_SIZE,
            dtype=DTYPE,
            device=DEVICE,
        )
        inputs.value_cache = torch.zeros_like(inputs.key_cache)
        run_reshape(inputs)
        if fused:
            tecoops.reshape_and_cache(
                inputs.key[-1:],
                inputs.value[-1:],
                inputs.slot_mapping[-1:],
                inputs.key_cache,
                inputs.value_cache,
            )
        run_attention(inputs)
        torch.sdaa.synchronize()
        output = inputs.out.cpu()
        del inputs
        release_device_cache()
        return output

    standalone_out = run_once(fused=False)
    fused_out = run_once(fused=True)
    difference = (standalone_out.float() - fused_out.float()).abs()
    exact = bool(torch.equal(standalone_out, fused_out))
    return {
        "name": case.name,
        "kind": "fused_vs_standalone_b1",
        "batch_size": 1,
        "max_kv_len": context_len,
        "bitwise_equal": exact,
        "max_abs_difference": float(difference.max()),
        "passed": exact,
    }


def measure_ms(operation, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        operation()
    torch.sdaa.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.sdaa.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def make_decode_inputs(context_len: int, seed: int) -> Inputs:
    # Include one new token so the benchmark mirrors a cache append followed by
    # attention over context_len + 1 tokens.
    return make_inputs(
        Case(f"decode_{context_len + 1}", (1,), (context_len + 1,)), seed
    )


def run_benchmark_case(
    case_id: str,
    context_len: int,
    seed: int,
    warmup: int,
    iterations: int,
    decode_iterations: int,
) -> dict:
    prefill = make_inputs(
        Case(f"{case_id}_prefill", (context_len,), (context_len,)), seed
    )
    reshape_metrics = measure_ms(
        lambda: run_reshape(prefill), warmup=warmup, iterations=iterations
    )
    prefill_metrics = measure_ms(
        lambda: run_attention(prefill), warmup=warmup, iterations=iterations
    )
    del prefill
    release_device_cache()

    decode = make_decode_inputs(context_len, seed + 1)
    run_reshape(decode)
    decode_reshape_metrics = measure_ms(
        lambda: tecoops.reshape_and_cache(
            decode.key[-1:],
            decode.value[-1:],
            decode.slot_mapping[-1:],
            decode.key_cache,
            decode.value_cache,
        ),
        warmup=warmup,
        iterations=decode_iterations,
    )
    decode_metrics = measure_ms(
        lambda: run_attention(decode),
        warmup=warmup,
        iterations=decode_iterations,
    )
    del decode
    release_device_cache()

    return {
        "case": case_id,
        "context_len": context_len,
        "output_len": 100,
        "reshape_full_context": reshape_metrics,
        "reshape_decode_single_token": decode_reshape_metrics,
        "prefill_attention": prefill_metrics,
        "decode_attention_single_step": decode_metrics,
    }


def release_device_cache() -> None:
    gc.collect()
    if hasattr(torch.sdaa, "empty_cache"):
        torch.sdaa.empty_cache()


def print_correctness(result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"[{status}] {result['name']}: batch={result['batch_size']} "
        f"q={result['total_q_tokens']} kv_max={result['max_kv_len']} "
        f"max_err={result['max_abs_error']:.6e} "
        f"cache={result['cache']['key_exact'] and result['cache']['value_exact']}"
    )


def print_replay_equivalence(result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"[{status}] {result['name']}: {result['kind']} "
        f"batch={result['batch_size']} kv_max={result['max_kv_len']} "
        f"bitwise={result['bitwise_equal']} "
        f"max_diff={result['max_abs_difference']:.6e}"
    )


def print_benchmark(result: dict) -> None:
    reshape = result["reshape_full_context"]["median_ms"]
    decode_reshape = result["reshape_decode_single_token"]["median_ms"]
    prefill = result["prefill_attention"]["median_ms"]
    decode = result["decode_attention_single_step"]["median_ms"]
    print(
        f"[{result['case']}] context={result['context_len']} "
        f"reshape={reshape:.3f} ms append-token={decode_reshape:.3f} ms "
        f"prefill={prefill:.3f} ms "
        f"decode-step={decode:.3f} ms"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("correctness", "benchmark", "all"), default="correctness"
    )
    parser.add_argument(
        "--suite",
        choices=("quick", "official", "all"),
        default="quick",
        help="quick covers Rice accuracy shapes; official covers T1-T4 prefill/decode",
    )
    parser.add_argument("--rows-per-sequence", type=int, default=5)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--decode-iterations", type=int, default=20)
    parser.add_argument(
        "--cases",
        default=",".join(OFFICIAL_CONTEXTS),
        help="comma-separated official cases, for example T1 or T1,T2",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cases = [
        item.strip().upper() for item in args.cases.split(",") if item.strip()
    ]
    unknown_cases = sorted(set(selected_cases) - set(OFFICIAL_CONTEXTS))
    if unknown_cases:
        raise ValueError(f"Unknown official cases: {', '.join(unknown_cases)}")

    if not torch.sdaa.is_available():
        raise RuntimeError("SDAA device is not available")

    report = {
        "model": "OneGenomeRice",
        "local_tp_shape": {
            "num_q_heads": NUM_Q_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "block_size": BLOCK_SIZE,
            "dtype": str(DTYPE),
        },
        "mode": args.mode,
        "suite": args.suite,
        "correctness": [],
        "kernel_replay_equivalence": [],
        "benchmark": [],
    }

    run_correctness = args.mode in ("correctness", "all")
    run_benchmark = args.mode in ("benchmark", "all")

    if run_correctness and args.suite in ("quick", "all"):
        print("\n=== OneGenomeRice correctness: accuracy workload ===")
        for index, case in enumerate(QUICK_CASES):
            result = run_correctness_case(
                case,
                seed=args.seed + index,
                rows_per_sequence=args.rows_per_sequence,
                atol=args.atol,
            )
            report["correctness"].append(result)
            print_correctness(result)

        print("\n=== Kernel identity: batched replay and fused decode ===")
        for index, case in enumerate(REPLAY_CASES):
            result = run_batched_replay_equivalence_case(
                case, seed=args.seed + 20 + index
            )
            report["kernel_replay_equivalence"].append(result)
            print_replay_equivalence(result)
        for index, context_len in enumerate(FUSED_REPLAY_CONTEXTS):
            result = run_fused_decode_equivalence_case(
                context_len, seed=args.seed + 40 + index
            )
            report["kernel_replay_equivalence"].append(result)
            print_replay_equivalence(result)

    if run_correctness and args.suite in ("official", "all"):
        print("\n=== OneGenomeRice correctness: T1-T4 long context ===")
        for index, case_id in enumerate(selected_cases):
            context_len = OFFICIAL_CONTEXTS[case_id]
            prefill_case = Case(
                f"{case_id}_prefill", (context_len,), (context_len,)
            )
            prefill_result = run_correctness_case(
                prefill_case,
                seed=args.seed + 100 + index,
                rows_per_sequence=args.rows_per_sequence,
                atol=args.atol,
            )
            report["correctness"].append(prefill_result)
            print_correctness(prefill_result)

            decode_case = Case(
                f"{case_id}_decode", (1,), (context_len,)
            )
            decode_result = run_correctness_case(
                decode_case,
                seed=args.seed + 200 + index,
                rows_per_sequence=1,
                atol=args.atol,
            )
            report["correctness"].append(decode_result)
            print_correctness(decode_result)

    if run_benchmark:
        print("\n=== OneGenomeRice microbenchmark: official T1-T4 ===")
        for index, case_id in enumerate(selected_cases):
            context_len = OFFICIAL_CONTEXTS[case_id]
            result = run_benchmark_case(
                case_id,
                context_len,
                seed=args.seed + 200 + index,
                warmup=args.warmup,
                iterations=args.iterations,
                decode_iterations=args.decode_iterations,
            )
            report["benchmark"].append(result)
            print_benchmark(result)

    checked_results = (
        report["correctness"] + report["kernel_replay_equivalence"]
    )
    report["passed"] = (
        all(item["passed"] for item in checked_results)
        if checked_results
        else None
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved report: {args.output}")
    else:
        print("\n" + rendered)

    if run_correctness and report["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
