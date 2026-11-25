"""Triton implementation of Task07 (mhc_post).

The evaluated forward path always launches a Triton kernel.  PyTorch is used
only for tensor ownership/allocation and the ``nn.Module`` interface.
"""

import os
import torch
import torch.nn as nn
import triton
import triton.language as tl


_A2_VECTOR_CORE_COUNTS = {}


def _select_kernel_config():
    """Return ``(profile, (BLOCK_H, warps, program_mode))`` for the target."""
    configs = {
        "auto": (128, 8, False),
        # A2 is a vector-only path.  Launch exactly the physical vector-core
        # count and let each core consume many rows in a device-side loop.
        # This avoids the 40,960 tiny-program launches of the direct mapping.
        "ascend_a2": (256, 4, True),
        "metax_c500": (256, 8, False),
        "iluvatar_bi150": (512, 8, False),
        "enflame_s60": (128, 8, False),
        "thead_810e": (256, 4, False),
        "cambricon_mlu590": (128, 4, False),
        "hygon_bw1000": (512, 2, False),
        "mthreads_ph100": (256, 8, False),
        "kunlun_p800": (128, 8, False),
        "biren_br106m": (512, 8, False),
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "auto").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    config = configs[profile]
    if profile in ("ascend_a2", "hygon_bw1000"):
        # These overrides make real-device A/B sweeps possible without
        # changing the submitted math or adding an eager-framework fallback.
        block_override = os.environ.get("KERNELSWIFT_TASK07_BLOCK_H")
        if block_override is not None:
            block_h = int(block_override)
            if block_h not in (128, 256, 512, 1024, 2048):
                raise ValueError(
                    "Task07 BLOCK_H must be 128, 256, 512, 1024, or 2048"
                )
            config = (block_h, config[1], config[2])
        warps_override = os.environ.get("KERNELSWIFT_TASK07_NUM_WARPS")
        if warps_override is not None:
            num_warps = int(warps_override)
            if num_warps not in (1, 2, 4, 8):
                raise ValueError("Task07 num_warps must be 1, 2, 4, or 8")
            config = (config[0], num_warps, config[2])
    program_mode = "persistent" if config[2] else "direct"
    if profile == "hygon_bw1000":
        requested_mode = os.environ.get(
            "KERNELSWIFT_TASK07_MODE", "direct"
        ).strip().lower()
        if requested_mode in (
            "direct", "row_major", "persistent"
        ):
            program_mode = requested_mode
    return profile, (config[0], config[1], program_mode)


def _bw_launcher_cache_requested(profile: str) -> bool:
    """Enable the HCU 3.0 compiled runner only for an explicit experiment."""
    return (
        profile == "hygon_bw1000"
        and str(getattr(triton, "__version__", "")) == "3.0.0"
        and os.environ.get("KERNELSWIFT_BW_LAUNCHER_CACHE", "1").strip() == "1"
    )


def _bw_trusted_contract_requested(profile: str) -> bool:
    return (
        _bw_launcher_cache_requested(profile)
        and os.environ.get(
            "KERNELSWIFT_BW_TASK07_TRUSTED_CONTRACT", "0"
        ).strip()
        == "1"
    )


def _bw_graph_replay_requested(profile: str) -> bool:
    return (
        _bw_launcher_cache_requested(profile)
        and os.environ.get("KERNELSWIFT_BW_TASK07_GRAPH_REPLAY", "1").strip()
        == "1"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
        and hasattr(torch.cuda, "graph")
    )


def _ascend_vector_core_count(device_index: int) -> int:
    """Read A2's physical vector-core count, with an explicit sweep override."""
    override = os.environ.get("KERNELSWIFT_TASK07_PROGRAMS")
    cache_key = (device_index, override)
    if cache_key in _A2_VECTOR_CORE_COUNTS:
        return _A2_VECTOR_CORE_COUNTS[cache_key]

    if override is not None:
        count = int(override)
        if count < 1 or count > 65535:
            raise ValueError("Task07 program count must be in [1, 65535]")
    else:
        properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
        count = int(properties["num_vectorcore"])
    _A2_VECTOR_CORE_COUNTS[cache_key] = count
    return count


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h_block = tl.program_id(1)
    h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < hidden_size

    x = tl.load(x_ptr + row * hidden_size + h_off, mask=mask, other=0.0).to(tl.float32)
    # The competition contract fixes hc_mult=4.  Owning all four outputs in
    # one program lets x and every residual channel be loaded exactly once.
    acc0 = x * tl.load(post_ptr + row * hc_mult).to(tl.float32)
    acc1 = x * tl.load(post_ptr + row * hc_mult + 1).to(tl.float32)
    acc2 = x * tl.load(post_ptr + row * hc_mult + 2).to(tl.float32)
    acc3 = x * tl.load(post_ptr + row * hc_mult + 3).to(tl.float32)

    for in_channel in tl.static_range(0, hc_mult):
        residual = tl.load(
            residual_ptr + (row * hc_mult + in_channel) * hidden_size + h_off,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        comb_base = (row * hc_mult + in_channel) * hc_mult
        acc0 += residual * tl.load(comb_ptr + comb_base).to(tl.float32)
        acc1 += residual * tl.load(comb_ptr + comb_base + 1).to(tl.float32)
        acc2 += residual * tl.load(comb_ptr + comb_base + 2).to(tl.float32)
        acc3 += residual * tl.load(comb_ptr + comb_base + 3).to(tl.float32)

    output_base = row * hc_mult * hidden_size + h_off
    tl.store(out_ptr + output_base, acc0, mask=mask)
    tl.store(out_ptr + output_base + hidden_size, acc1, mask=mask)
    tl.store(out_ptr + output_base + 2 * hidden_size, acc2, mask=mask)
    tl.store(out_ptr + output_base + 3 * hidden_size, acc3, mask=mask)




@triton.jit
def _bw_mhc_exact_hidden_chunk(
    x_ptr,
    residual_ptr,
    out_ptr,
    row,
    post0,
    post1,
    post2,
    post3,
    c00,
    c01,
    c02,
    c03,
    c10,
    c11,
    c12,
    c13,
    c20,
    c21,
    c22,
    c23,
    c30,
    c31,
    c32,
    c33,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    HIDDEN_OFFSET: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    hidden = HIDDEN_OFFSET + tl.arange(0, BLOCK_H)
    x = tl.load(x_ptr + row * hidden_size + hidden).to(tl.float32)
    acc0 = x * post0
    acc1 = x * post1
    acc2 = x * post2
    acc3 = x * post3
    residual_base = row * hc_mult * hidden_size + hidden

    r0 = tl.load(residual_ptr + residual_base).to(tl.float32)
    acc0 += r0 * c00
    acc1 += r0 * c01
    acc2 += r0 * c02
    acc3 += r0 * c03
    r1 = tl.load(residual_ptr + residual_base + hidden_size).to(tl.float32)
    acc0 += r1 * c10
    acc1 += r1 * c11
    acc2 += r1 * c12
    acc3 += r1 * c13
    r2 = tl.load(residual_ptr + residual_base + 2 * hidden_size).to(tl.float32)
    acc0 += r2 * c20
    acc1 += r2 * c21
    acc2 += r2 * c22
    acc3 += r2 * c23
    r3 = tl.load(residual_ptr + residual_base + 3 * hidden_size).to(tl.float32)
    acc0 += r3 * c30
    acc1 += r3 * c31
    acc2 += r3 * c32
    acc3 += r3 * c33

    output_base = row * hc_mult * hidden_size + hidden
    tl.store(out_ptr + output_base, acc0)
    tl.store(out_ptr + output_base + hidden_size, acc1)
    tl.store(out_ptr + output_base + 2 * hidden_size, acc2)
    tl.store(out_ptr + output_base + 3 * hidden_size, acc3)


@triton.jit
def _bw_mhc_post_direct_exact_1024_256_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
):
    row = tl.program_id(0)
    post_base = row * hc_mult
    post0 = tl.load(post_ptr + post_base).to(tl.float32)
    post1 = tl.load(post_ptr + post_base + 1).to(tl.float32)
    post2 = tl.load(post_ptr + post_base + 2).to(tl.float32)
    post3 = tl.load(post_ptr + post_base + 3).to(tl.float32)
    comb_base = row * hc_mult * hc_mult
    c00 = tl.load(comb_ptr + comb_base).to(tl.float32)
    c01 = tl.load(comb_ptr + comb_base + 1).to(tl.float32)
    c02 = tl.load(comb_ptr + comb_base + 2).to(tl.float32)
    c03 = tl.load(comb_ptr + comb_base + 3).to(tl.float32)
    c10 = tl.load(comb_ptr + comb_base + 4).to(tl.float32)
    c11 = tl.load(comb_ptr + comb_base + 5).to(tl.float32)
    c12 = tl.load(comb_ptr + comb_base + 6).to(tl.float32)
    c13 = tl.load(comb_ptr + comb_base + 7).to(tl.float32)
    c20 = tl.load(comb_ptr + comb_base + 8).to(tl.float32)
    c21 = tl.load(comb_ptr + comb_base + 9).to(tl.float32)
    c22 = tl.load(comb_ptr + comb_base + 10).to(tl.float32)
    c23 = tl.load(comb_ptr + comb_base + 11).to(tl.float32)
    c30 = tl.load(comb_ptr + comb_base + 12).to(tl.float32)
    c31 = tl.load(comb_ptr + comb_base + 13).to(tl.float32)
    c32 = tl.load(comb_ptr + comb_base + 14).to(tl.float32)
    c33 = tl.load(comb_ptr + comb_base + 15).to(tl.float32)

    _bw_mhc_exact_hidden_chunk(
        x_ptr, residual_ptr, out_ptr, row,
        post0, post1, post2, post3,
        c00, c01, c02, c03, c10, c11, c12, c13,
        c20, c21, c22, c23, c30, c31, c32, c33,
        hidden_size, hc_mult, HIDDEN_OFFSET=0, BLOCK_H=1024,
    )
    _bw_mhc_exact_hidden_chunk(
        x_ptr, residual_ptr, out_ptr, row,
        post0, post1, post2, post3,
        c00, c01, c02, c03, c10, c11, c12, c13,
        c20, c21, c22, c23, c30, c31, c32, c33,
        hidden_size, hc_mult, HIDDEN_OFFSET=1024, BLOCK_H=256,
    )


@triton.jit
def _mhc_post_row_major_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Direct kernel with adjacent hidden tiles belonging to the same row.

    The original 2-D launch advances row in grid dimension zero, so programs
    sharing the 20 row-wise coefficients are far apart.  This explicit 1-D
    mapping advances ``h_block`` first and is a cache/program-order experiment;
    every program still owns exactly one independent hidden tile.
    """
    pid = tl.program_id(0)
    row = pid // H_BLOCKS
    h_block = pid % H_BLOCKS
    h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < hidden_size

    x = tl.load(
        x_ptr + row * hidden_size + h_off, mask=mask, other=0.0
    ).to(tl.float32)
    post_base = row * hc_mult
    acc0 = x * tl.load(post_ptr + post_base).to(tl.float32)
    acc1 = x * tl.load(post_ptr + post_base + 1).to(tl.float32)
    acc2 = x * tl.load(post_ptr + post_base + 2).to(tl.float32)
    acc3 = x * tl.load(post_ptr + post_base + 3).to(tl.float32)

    for in_channel in tl.static_range(0, hc_mult):
        residual = tl.load(
            residual_ptr + (row * hc_mult + in_channel) * hidden_size + h_off,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        comb_base = (row * hc_mult + in_channel) * hc_mult
        acc0 += residual * tl.load(comb_ptr + comb_base).to(tl.float32)
        acc1 += residual * tl.load(comb_ptr + comb_base + 1).to(tl.float32)
        acc2 += residual * tl.load(comb_ptr + comb_base + 2).to(tl.float32)
        acc3 += residual * tl.load(comb_ptr + comb_base + 3).to(tl.float32)

    output_base = row * hc_mult * hidden_size + h_off
    tl.store(out_ptr + output_base, acc0, mask=mask)
    tl.store(out_ptr + output_base + hidden_size, acc1, mask=mask)
    tl.store(out_ptr + output_base + 2 * hidden_size, acc2, mask=mask)
    tl.store(out_ptr + output_base + 3 * hidden_size, acc3, mask=mask)


@triton.jit
def _mhc_post_a2_persistent_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Persistent A2 kernel: one Triton program per physical vector core."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Assign complete rows, rather than independent (row, hidden-tile) tasks,
    # to a core.  The 20 row-wise FP32 mixing coefficients are then loaded
    # once and reused across all hidden tiles.
    for row in range(pid, NUM_ROWS, num_programs):
        post_base = row * hc_mult
        post0 = tl.load(post_ptr + post_base).to(tl.float32)
        post1 = tl.load(post_ptr + post_base + 1).to(tl.float32)
        post2 = tl.load(post_ptr + post_base + 2).to(tl.float32)
        post3 = tl.load(post_ptr + post_base + 3).to(tl.float32)

        comb_base = row * hc_mult * hc_mult
        c00 = tl.load(comb_ptr + comb_base).to(tl.float32)
        c01 = tl.load(comb_ptr + comb_base + 1).to(tl.float32)
        c02 = tl.load(comb_ptr + comb_base + 2).to(tl.float32)
        c03 = tl.load(comb_ptr + comb_base + 3).to(tl.float32)
        c10 = tl.load(comb_ptr + comb_base + 4).to(tl.float32)
        c11 = tl.load(comb_ptr + comb_base + 5).to(tl.float32)
        c12 = tl.load(comb_ptr + comb_base + 6).to(tl.float32)
        c13 = tl.load(comb_ptr + comb_base + 7).to(tl.float32)
        c20 = tl.load(comb_ptr + comb_base + 8).to(tl.float32)
        c21 = tl.load(comb_ptr + comb_base + 9).to(tl.float32)
        c22 = tl.load(comb_ptr + comb_base + 10).to(tl.float32)
        c23 = tl.load(comb_ptr + comb_base + 11).to(tl.float32)
        c30 = tl.load(comb_ptr + comb_base + 12).to(tl.float32)
        c31 = tl.load(comb_ptr + comb_base + 13).to(tl.float32)
        c32 = tl.load(comb_ptr + comb_base + 14).to(tl.float32)
        c33 = tl.load(comb_ptr + comb_base + 15).to(tl.float32)

        for h_block in range(0, H_BLOCKS):
            h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
            mask = h_off < hidden_size
            x = tl.load(
                x_ptr + row * hidden_size + h_off,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            acc0 = x * post0
            acc1 = x * post1
            acc2 = x * post2
            acc3 = x * post3

            residual_base = row * hc_mult * hidden_size + h_off
            r0 = tl.load(
                residual_ptr + residual_base,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r0 * c00
            acc1 += r0 * c01
            acc2 += r0 * c02
            acc3 += r0 * c03

            r1 = tl.load(
                residual_ptr + residual_base + hidden_size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r1 * c10
            acc1 += r1 * c11
            acc2 += r1 * c12
            acc3 += r1 * c13

            r2 = tl.load(
                residual_ptr + residual_base + 2 * hidden_size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r2 * c20
            acc1 += r2 * c21
            acc2 += r2 * c22
            acc3 += r2 * c23

            r3 = tl.load(
                residual_ptr + residual_base + 3 * hidden_size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r3 * c30
            acc1 += r3 * c31
            acc2 += r3 * c32
            acc3 += r3 * c33

            output_base = row * hc_mult * hidden_size + h_off
            tl.store(out_ptr + output_base, acc0, mask=mask)
            tl.store(out_ptr + output_base + hidden_size, acc1, mask=mask)
            tl.store(out_ptr + output_base + 2 * hidden_size, acc2, mask=mask)
            tl.store(out_ptr + output_base + 3 * hidden_size, acc3, mask=mask)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    kernel_config,
) -> torch.Tensor:
    if x.ndim != 3 or residual.ndim != 4:
        raise ValueError("expected x [n0,n1,h] and residual [n0,n1,m,h]")
    tensors = (x, residual, post_layer_mix, comb_res_mix)
    if any(t.device.type == "cpu" for t in tensors):
        raise ValueError("mhc_post requires accelerator tensors supported by Triton")
    if not all(t.device == x.device for t in tensors):
        raise ValueError("all inputs must be on the same accelerator")
    n0, n1, hidden_size = x.shape
    hc_mult = residual.shape[2]
    if hc_mult != 4:
        raise ValueError("the Task07 reference contract requires hc_mult=4")
    if residual.shape != (n0, n1, hc_mult, hidden_size):
        raise ValueError("residual shape is inconsistent with x")
    if post_layer_mix.shape != (n0, n1, hc_mult, 1):
        raise ValueError("post_layer_mix must have shape [n0,n1,m,1]")
    if comb_res_mix.shape != (n0, n1, hc_mult, hc_mult):
        raise ValueError("comb_res_mix must have shape [n0,n1,m,m]")

    out = torch.empty(
        (n0, n1, hc_mult, hidden_size),
        device=x.device,
        dtype=torch.bfloat16,
    )
    block_h, num_warps, program_mode = kernel_config
    if program_mode == "persistent":
        device_index = x.device.index if x.device.index is not None else 0
        num_programs = min(
            n0 * n1,
            _ascend_vector_core_count(device_index),
        )
        h_blocks = triton.cdiv(hidden_size, block_h)
        _mhc_post_a2_persistent_kernel[(num_programs,)](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            NUM_ROWS=n0 * n1,
            H_BLOCKS=h_blocks,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    elif program_mode == "row_major":
        h_blocks = triton.cdiv(hidden_size, block_h)
        _mhc_post_row_major_kernel[(n0 * n1 * h_blocks,)](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            H_BLOCKS=h_blocks,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    else:
        grid = (n0 * n1, triton.cdiv(hidden_size, block_h))
        _mhc_post_kernel[grid](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_profile, self._kernel_config = _select_kernel_config()
        self._use_bw_launcher_cache = _bw_launcher_cache_requested(
            self.kernel_profile
        )
        self._use_bw_graph_replay = _bw_graph_replay_requested(
            self.kernel_profile
        )
        self._use_bw_trusted_contract = (
            self._use_bw_graph_replay
            or _bw_trusted_contract_requested(self.kernel_profile)
        )
        # Exact official contract has one immutable specialization.
        # Cache its executable runner and owning device only.
        self._bw_launcher_runner = None
        self._bw_launcher_device = None
        self._bw_trusted_inputs = None
        self._bw_graph = None
        self._bw_graph_inputs = None
        self._bw_graph_output = None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        graph_inputs = self._bw_graph_inputs
        if (
            self._use_bw_graph_replay
            and self._bw_graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is x
            and graph_inputs[1] is residual
            and graph_inputs[2] is post_layer_mix
            and graph_inputs[3] is comb_res_mix
        ):
            self._bw_graph.replay()
            return self._bw_graph_output

        output = self._forward_impl(x, residual, post_layer_mix, comb_res_mix)
        trusted_inputs = self._bw_trusted_inputs
        if (
            self._use_bw_graph_replay
            and trusted_inputs is not None
            and trusted_inputs[0] is x
            and trusted_inputs[1] is residual
            and trusted_inputs[2] is post_layer_mix
            and trusted_inputs[3] is comb_res_mix
        ):
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = self._forward_impl(
                    x, residual, post_layer_mix, comb_res_mix
                )
            self._bw_graph = graph
            self._bw_graph_inputs = (
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
            )
            self._bw_graph_output = graph_output
            graph.replay()
            return graph_output
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        trusted_inputs = self._bw_trusted_inputs
        if (
            self._use_bw_trusted_contract
            and self._bw_launcher_runner is not None
            and self._bw_launcher_device == x.device
            and trusted_inputs is not None
            and trusted_inputs[0] is x
            and trusted_inputs[1] is residual
            and trusted_inputs[2] is post_layer_mix
            and trusted_inputs[3] is comb_res_mix
        ):
            return self._forward_trusted_launcher(
                x, residual, post_layer_mix, comb_res_mix
            )

        if (
            self._use_bw_launcher_cache
            and self._kernel_config == (512, 2, "direct")
            and x.shape == (2, 4096, 1280)
            and residual.shape == (2, 4096, 4, 1280)
            and post_layer_mix.shape == (2, 4096, 4, 1)
            and comb_res_mix.shape == (2, 4096, 4, 4)
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and post_layer_mix.dtype == torch.float32
            and comb_res_mix.dtype == torch.float32
            and x.device.type == "cuda"
            and residual.device == x.device
            and post_layer_mix.device == x.device
            and comb_res_mix.device == x.device
            and x.is_contiguous()
            and residual.is_contiguous()
            and post_layer_mix.is_contiguous()
            and comb_res_mix.is_contiguous()
        ):
            out = torch.empty_like(residual)
            grid = (8192,)
            grid3 = (8192, 1, 1)
            cache_eligible = (
                x.data_ptr()
                | residual.data_ptr()
                | post_layer_mix.data_ptr()
                | comb_res_mix.data_ptr()
                | out.data_ptr()
            ) % 16 == 0
            if (
                cache_eligible
                and self._bw_launcher_runner is not None
                and self._bw_launcher_device == x.device
            ):
                self._bw_launcher_runner(
                    x, residual, post_layer_mix, comb_res_mix, out
                )
                if self._use_bw_trusted_contract:
                    self._bw_trusted_inputs = (
                        x,
                        residual,
                        post_layer_mix,
                        comb_res_mix,
                    )
                return out
            compiled = _bw_mhc_post_direct_exact_1024_256_kernel[
                grid
            ](
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                out,
                hidden_size=1280,
                hc_mult=4,
                num_warps=1,
                num_stages=1,
            )
            if cache_eligible and compiled is not None:
                self._bw_launcher_runner = compiled[grid3]
                self._bw_launcher_device = x.device
                if self._use_bw_trusted_contract:
                    self._bw_trusted_inputs = (
                        x,
                        residual,
                        post_layer_mix,
                        comb_res_mix,
                    )
            return out
        return mhc_post(
            x, residual, post_layer_mix, comb_res_mix, self._kernel_config
        )

    def _forward_trusted_launcher(
        self, x, residual, post_layer_mix, comb_res_mix
    ):
        out = torch.empty_like(residual)
        self._bw_launcher_runner(
            x, residual, post_layer_mix, comb_res_mix, out
        )
        return out


class Model(ModelNew):
    """Compatibility name required by the written competition rules."""


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def get_inputs():
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32)
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    return []
