"""Numerical varlen/custom-op regression across two processes and a disk cache.

Inputs, prefix sums, reference results and assertions remain on the GPU.
The second process must load the first process's objects without retuning.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the frozen varlen runtime test requires SM90",
)


def _run_varlen_case(leaf: str, rows: int) -> None:
    from cute_gemm.gemm_interface import gemm

    torch.manual_seed(1024)
    torch.backends.cuda.matmul.allow_tf32 = False
    experts, width = 32, 512
    cu_seqlens = torch.arange(experts + 1, device="cuda", dtype=torch.int32) * rows
    if leaf.startswith("m-"):
        A = torch.randint(-2, 3, (experts * rows, width), device="cuda").to(torch.bfloat16)
        weights = torch.randint(-2, 3, (experts, width, width), device="cuda").to(torch.bfloat16)
        output = gemm(A, weights.mT, cu_seqlens_m=cu_seqlens)
        reference = torch.bmm(A.reshape(experts, rows, width).float(), weights.mT.float())
        reference = reference.reshape(experts * rows, width)
    else:
        A = torch.randint(-2, 3, (experts * rows, 128), device="cuda").to(torch.bfloat16)
        B = torch.randint(-2, 3, (experts * rows, width), device="cuda").to(torch.bfloat16)
        output = gemm(A.T, B, cu_seqlens_k=cu_seqlens)
        reference = torch.bmm(
            A.reshape(experts, rows, 128).mT.float(),
            B.reshape(experts, rows, width).float(),
        )
    # Integer inputs have exactly representable fp32 sums at these sizes;
    # the kernel and reference must agree after bf16 rounding.  Do not use
    # allclose()/assert_close(): returning a host bool would read back data.
    torch._assert_async(
        torch.isfinite(output).all() & (output == reference.to(torch.bfloat16)).all(),
        f"{leaf}: varlen output differs from the fp32 reference",
    )
    torch.cuda.synchronize()


@requires_sm90
@pytest.mark.parametrize("leaf,rows", [("m-low", 64), ("k-high", 2048)])
def test_varlen_custom_op_numerics_and_second_process_cache(tmp_path, leaf, rows):
    env = {
        **os.environ,
        "DLKERNEL_CACHE_DIR": str(tmp_path / "dlkernel"),
        "DLKERNEL_ARTIFACT_CACHE_DIR": str(tmp_path / "dlkernel-artifacts"),
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        "CUTE_DSL_CACHE_DIR": str(tmp_path / "cute"),
        "DLKERNEL_CACHE_ENABLED": "1",
        "DLKERNEL_CACHE_AUTOTUNING": "1",
        "DLKERNEL_VARLEN_SELECTOR": "1",
        "DLKERNEL_TUNE_WARMUP_MS": "50",
        "DLKERNEL_TUNE_TIMED_CALLS": "64",
        "DLKERNEL_DEBUG_TUNE": "1",
    }
    for name in ("DLKERNEL_ARCH", "CUTE_DSL_ARCH", "DLKERNEL_FORCE_CACHE_UPDATE"):
        env.pop(name, None)
    command = [sys.executable, str(Path(__file__).resolve()), leaf, str(rows)]
    env["DLKERNEL_REQUIRE_TUNE_CACHE"] = "0"
    cold = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert cold.returncode == 0, cold.stdout + cold.stderr
    assert "DLKERNEL_TUNE_START" in cold.stdout and "configs=8" in cold.stdout
    assert f"rule={leaf}" in cold.stdout
    objects = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.o")}
    assert objects, "The first process must publish compiled objects"

    env["DLKERNEL_REQUIRE_TUNE_CACHE"] = "1"
    warm = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert warm.returncode == 0, warm.stdout + warm.stderr
    assert "DLKERNEL_TUNE_DISK_HIT" in warm.stdout
    assert "DLKERNEL_TUNE_START" not in warm.stdout
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.o")} == objects


if __name__ == "__main__":
    _run_varlen_case(sys.argv[1], int(sys.argv[2]))
