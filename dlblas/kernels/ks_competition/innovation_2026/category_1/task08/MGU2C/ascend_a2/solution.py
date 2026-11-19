"""Fused Triton implementation of hc_split_sinkhorn with A/B launch variants."""

import os
from importlib import metadata as importlib_metadata

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.compiler.compiler import CompiledKernel


def _is_triton_ascend_321_distribution():
    """Admit the private CompiledKernel ABI only on audited 3.2.1 files."""
    distribution = next(
        (
            item
            for item in importlib_metadata.distributions()
            if item.metadata.get("Name", "").lower().replace("_", "-")
            == "triton-ascend"
        ),
        None,
    )
    if distribution is None:
        return False
    version = distribution.version.split("+", 1)[0]
    distribution_triton = os.path.realpath(
        str(distribution.locate_file("triton"))
    )
    imported_triton = os.path.realpath(str(triton.__file__))
    common_path = os.path.commonpath(
        (distribution_triton, imported_triton)
    )
    return (
        version == "3.2.1"
        and common_path == distribution_triton
        and callable(getattr(CompiledKernel, "__getitem__", None))
    )


def _compiled_grid_runner(compiled, grid3):
    """Extract a v3.2.1 grid runner without retaining call-time tensors."""
    if not isinstance(compiled, CompiledKernel):
        return None
    runner = compiled[grid3]
    return runner if callable(runner) else None


def _matches_frozen_sinkhorn_inputs(tensors):
    """Check all pointer specializations in Task08's official input ABI."""
    specs = (
        (torch.float32, (2, 8, 24), (192, 24, 1)),
        (torch.float32, (3,), (1,)),
        (torch.float32, (24,), (1,)),
    )
    if len(tensors) != len(specs):
        return False
    first_device = tensors[0].device
    if first_device.type != "npu":
        return False
    for tensor, (dtype, shape, stride) in zip(tensors, specs):
        if (
            tensor.device != first_device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != stride
            or tensor.data_ptr() % 16 != 0
        ):
            return False
    return True


def _select_kernel_config():
    """Return the profile and explicitly selectable 4x4 kernel controls."""
    configs = {
        "auto": 1,
        "ascend_a2": 2,
        "metax_c500": 1,
        "iluvatar_bi150": 2,
        "enflame_s60": 4,
        "thead_810e": 1,
        "cambricon_mlu590": 2,
        "hygon_bw1000": 4,
        "mthreads_ph100": 1,
        "kunlun_p800": 2,
        "biren_br106m": 4,
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "ascend_a2").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    warp_overrides = {"1": 1, "2": 2, "4": 4, "8": 8}
    num_warps = warp_overrides.get(
        os.environ.get("KERNELSWIFT_TASK08_NUM_WARPS", "").strip(),
        configs[profile],
    )
    sinkhorn_mode = os.environ.get(
        "KERNELSWIFT_TASK08_SINKHORN", "direct"
    ).strip().lower()
    use_factorized_sinkhorn = sinkhorn_mode == "factorized"
    rows_per_program = {
        "1": 1,
        "2": 2,
        "4": 4,
        "8": 8,
        "16": 16,
    }.get(os.environ.get("KERNELSWIFT_TASK08_ROWS_PER_PROGRAM", "1").strip(), 1)
    unroll_sinkhorn = os.environ.get(
        "KERNELSWIFT_TASK08_UNROLL", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    return (
        profile,
        num_warps,
        use_factorized_sinkhorn,
        rows_per_program,
        unroll_sinkhorn,
    )


@triton.jit
def _hc_split_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_out_ptr,
    hc: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    eps: tl.constexpr,
    FACTORIZED_SINKHORN: tl.constexpr,
    UNROLL_SINKHORN: tl.constexpr,
):
    row_id = tl.program_id(0)
    mix_width: tl.constexpr = (2 + hc) * hc
    v = tl.arange(0, hc)
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre_x = tl.load(mixes_ptr + row_id * mix_width + v)
    pre_b = tl.load(base_ptr + v)
    pre_z = pre_x * s0 + pre_b
    pre = 1.0 / (1.0 + tl.exp(-pre_z)) + eps
    tl.store(pre_ptr + row_id * hc + v, pre)

    post_x = tl.load(mixes_ptr + row_id * mix_width + hc + v)
    post_b = tl.load(base_ptr + hc + v)
    post_z = post_x * s1 + post_b
    post = 2.0 / (1.0 + tl.exp(-post_z))
    tl.store(post_ptr + row_id * hc + v, post)

    r = tl.arange(0, hc)[:, None]
    c = tl.arange(0, hc)[None, :]
    matrix_off = r * hc + c
    raw = tl.load(mixes_ptr + row_id * mix_width + 2 * hc + matrix_off)
    comb = raw * s2 + tl.load(base_ptr + 2 * hc + matrix_off)

    row_max = tl.max(comb, axis=1)[:, None]
    comb = tl.exp(comb - row_max)
    row_sum = tl.sum(comb, axis=1)[:, None]
    comb = comb / row_sum + eps
    if FACTORIZED_SINKHORN:
        # Keep the row/column scaling vectors instead of dividing all sixteen
        # matrix entries every half-iteration.  If
        #
        #   comb[i, j] = kernel[i, j] * row_scale[i] * col_scale[j],
        #
        # then updating a scale by its current row/column denominator is
        # algebraically identical to materialising and normalising ``comb``.
        # This replaces 32 scalar divisions per Sinkhorn round with 8
        # divisions and inexpensive multiplies; the final matrix is formed
        # only once.  The additive eps terms are retained exactly.
        kernel = comb
        row_scale = tl.full((hc, 1), 1.0, tl.float32)
        col_sum = tl.sum(kernel, axis=0)[None, :]
        col_scale = 1.0 / (col_sum + eps)
        if UNROLL_SINKHORN:
            for _ in range(sinkhorn_iters - 1):
                row_sum = tl.sum(kernel * col_scale, axis=1)[:, None]
                row_scale = row_scale / (row_scale * row_sum + eps)
                col_sum = tl.sum(kernel * row_scale, axis=0)[None, :]
                col_scale = col_scale / (col_scale * col_sum + eps)
        else:
            for _ in tl.range(0, sinkhorn_iters - 1):
                row_sum = tl.sum(kernel * col_scale, axis=1)[:, None]
                row_scale = row_scale / (row_scale * row_sum + eps)
                col_sum = tl.sum(kernel * row_scale, axis=0)[None, :]
                col_scale = col_scale / (col_scale * col_sum + eps)
        comb = kernel * row_scale * col_scale
    else:
        col_sum = tl.sum(comb, axis=0)[None, :]
        comb = comb / (col_sum + eps)

        # Keep this as a compiler loop.  Fully unrolling 19 matrix-reduction
        # rounds causes excessive compile time on several Triton backends.
        if UNROLL_SINKHORN:
            for _ in range(sinkhorn_iters - 1):
                row_sum = tl.sum(comb, axis=1)[:, None]
                comb = comb / (row_sum + eps)
                col_sum = tl.sum(comb, axis=0)[None, :]
                comb = comb / (col_sum + eps)
        else:
            for _ in tl.range(0, sinkhorn_iters - 1):
                row_sum = tl.sum(comb, axis=1)[:, None]
                comb = comb / (row_sum + eps)
                col_sum = tl.sum(comb, axis=0)[None, :]
                comb = comb / (col_sum + eps)

    tl.store(comb_out_ptr + row_id * hc * hc + matrix_off, comb)


@triton.jit
def _hc_split_sinkhorn_rows_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_out_ptr,
    num_rows,
    hc: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    eps: tl.constexpr,
    FACTORIZED_SINKHORN: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    UNROLL_SINKHORN: tl.constexpr,
):
    """Process several independent 4x4 problems in one Triton program."""
    row_lane = tl.arange(0, ROWS_PER_PROGRAM)
    row_id = tl.program_id(0) * ROWS_PER_PROGRAM + row_lane
    valid_row = row_id < num_rows
    mix_width: tl.constexpr = (2 + hc) * hc
    v = tl.arange(0, hc)
    vector_off = row_id[:, None] * mix_width + v[None, :]
    vector_mask = valid_row[:, None]
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre_x = tl.load(mixes_ptr + vector_off, mask=vector_mask, other=0.0)
    pre_b = tl.load(base_ptr + v)
    pre_z = pre_x * s0 + pre_b[None, :]
    pre = 1.0 / (1.0 + tl.exp(-pre_z)) + eps
    vector_lane = tl.arange(0, ROWS_PER_PROGRAM * hc)
    vector_base = tl.program_id(0) * ROWS_PER_PROGRAM * hc
    vector_flat_mask = vector_base + vector_lane < num_rows * hc
    # Flatten consecutive row owners before GM writes.  For two or more rows
    # the final axis is at least eight FP32 values (32 bytes), matching A2's
    # minimum efficient Vector/UB transfer granularity.
    tl.store(
        pre_ptr + vector_base + vector_lane,
        tl.reshape(pre, (ROWS_PER_PROGRAM * hc,)),
        mask=vector_flat_mask,
    )

    post_x = tl.load(mixes_ptr + vector_off + hc, mask=vector_mask, other=0.0)
    post_b = tl.load(base_ptr + hc + v)
    post_z = post_x * s1 + post_b[None, :]
    post = 2.0 / (1.0 + tl.exp(-post_z))
    tl.store(
        post_ptr + vector_base + vector_lane,
        tl.reshape(post, (ROWS_PER_PROGRAM * hc,)),
        mask=vector_flat_mask,
    )

    matrix_row = tl.arange(0, hc)[None, :, None]
    matrix_col = tl.arange(0, hc)[None, None, :]
    matrix_off = matrix_row * hc + matrix_col
    matrix_ptrs = (
        mixes_ptr
        + row_id[:, None, None] * mix_width
        + 2 * hc
        + matrix_off
    )
    matrix_mask = valid_row[:, None, None]
    raw = tl.load(matrix_ptrs, mask=matrix_mask, other=0.0)
    base = tl.load(base_ptr + 2 * hc + matrix_off)
    comb = raw * s2 + base

    row_max = tl.max(comb, axis=2)[:, :, None]
    comb = tl.exp(comb - row_max)
    row_sum = tl.sum(comb, axis=2)[:, :, None]
    # Preserve the reference insertion point: eps is added after the initial
    # row softmax, before the first column normalization.
    comb = comb / row_sum + eps
    if FACTORIZED_SINKHORN:
        kernel = comb
        row_scale = tl.full((ROWS_PER_PROGRAM, hc, 1), 1.0, tl.float32)
        col_sum = tl.sum(kernel, axis=1)[:, None, :]
        col_scale = 1.0 / (col_sum + eps)
        if UNROLL_SINKHORN:
            for _ in range(sinkhorn_iters - 1):
                row_sum = tl.sum(kernel * col_scale, axis=2)[:, :, None]
                row_scale = row_scale / (row_scale * row_sum + eps)
                col_sum = tl.sum(kernel * row_scale, axis=1)[:, None, :]
                col_scale = col_scale / (col_scale * col_sum + eps)
        else:
            for _ in tl.range(0, sinkhorn_iters - 1):
                row_sum = tl.sum(kernel * col_scale, axis=2)[:, :, None]
                row_scale = row_scale / (row_scale * row_sum + eps)
                col_sum = tl.sum(kernel * row_scale, axis=1)[:, None, :]
                col_scale = col_scale / (col_scale * col_sum + eps)
        comb = kernel * row_scale * col_scale
    else:
        col_sum = tl.sum(comb, axis=1)[:, None, :]
        comb = comb / (col_sum + eps)
        if UNROLL_SINKHORN:
            for _ in range(sinkhorn_iters - 1):
                row_sum = tl.sum(comb, axis=2)[:, :, None]
                comb = comb / (row_sum + eps)
                col_sum = tl.sum(comb, axis=1)[:, None, :]
                comb = comb / (col_sum + eps)
        else:
            for _ in tl.range(0, sinkhorn_iters - 1):
                row_sum = tl.sum(comb, axis=2)[:, :, None]
                comb = comb / (row_sum + eps)
                col_sum = tl.sum(comb, axis=1)[:, None, :]
                comb = comb / (col_sum + eps)

    matrix_lane = tl.arange(0, ROWS_PER_PROGRAM * hc * hc)
    matrix_base = tl.program_id(0) * ROWS_PER_PROGRAM * hc * hc
    matrix_flat_mask = matrix_base + matrix_lane < num_rows * hc * hc
    tl.store(
        comb_out_ptr + matrix_base + matrix_lane,
        tl.reshape(comb, (ROWS_PER_PROGRAM * hc * hc,)),
        mask=matrix_flat_mask,
    )


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    num_warps: int,
    use_factorized_sinkhorn: bool,
    rows_per_program: int = 1,
    unroll_sinkhorn: bool = False,
    pack_outputs: bool = False,
    packed_alignment: int = 1,
):
    tensors = (mixes, hc_scale, hc_base)
    if any(t.device.type == "cpu" for t in tensors):
        raise ValueError("hc_split_sinkhorn requires accelerator tensors supported by Triton")
    if not all(t.device == mixes.device for t in tensors):
        raise ValueError("all inputs must be on the same accelerator")
    b, s, mix_hc = mixes.shape
    expected = (2 + hc_mult) * hc_mult
    if mix_hc != expected:
        raise ValueError(f"expected mix dim {expected}, got {mix_hc}")
    num_rows = b * s
    if pack_outputs:
        # One fresh allocation, split into three non-overlapping contiguous
        # views.  This only removes allocator bookkeeping; no output storage is
        # retained between forwards and every kernel invocation still writes
        # all values.
        pre_elements = num_rows * hc_mult
        comb_elements = num_rows * hc_mult * hc_mult
        post_offset = triton.cdiv(pre_elements, packed_alignment) * packed_alignment
        comb_offset = post_offset + post_offset
        storage = torch.empty(
            comb_offset + comb_elements,
            device=mixes.device,
            dtype=torch.float32,
        )
        pre = storage[:pre_elements].view(b, s, hc_mult)
        post = storage[post_offset : post_offset + pre_elements].view(
            b, s, hc_mult
        )
        comb = storage[comb_offset : comb_offset + comb_elements].view(
            b, s, hc_mult, hc_mult
        )
    else:
        pre = torch.empty((b, s, hc_mult), device=mixes.device, dtype=torch.float32)
        post = torch.empty_like(pre)
        comb = torch.empty(
            (b, s, hc_mult, hc_mult), device=mixes.device, dtype=torch.float32
        )
    if rows_per_program == 1:
        _hc_split_sinkhorn_kernel[(num_rows,)](
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            hc=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            FACTORIZED_SINKHORN=use_factorized_sinkhorn,
            UNROLL_SINKHORN=unroll_sinkhorn,
            num_warps=num_warps,
        )
    else:
        _hc_split_sinkhorn_rows_kernel[(triton.cdiv(num_rows, rows_per_program),)](
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            num_rows,
            hc=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            FACTORIZED_SINKHORN=use_factorized_sinkhorn,
            ROWS_PER_PROGRAM=rows_per_program,
            UNROLL_SINKHORN=unroll_sinkhorn,
            num_warps=num_warps,
        )
    return pre, post, comb


class ModelNew(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        (
            self.kernel_profile,
            self._num_warps,
            self._use_factorized_sinkhorn,
            self._rows_per_program,
            self._unroll_sinkhorn,
        ) = _select_kernel_config()
        output_packing = os.environ.get(
            "KERNELSWIFT_TASK08_OUTPUT_PACKING", "separate"
        ).strip().lower()
        self._pack_outputs = output_packing in ("packed", "packed512")
        self._packed_alignment = 128 if output_packing == "packed512" else 1
        launch_mode = os.environ.get(
            "KERNELSWIFT_TASK08_LAUNCH_MODE", "compiled"
        ).strip().lower()
        self._use_compiled_launcher = (
            launch_mode == "compiled"
            and self.kernel_profile == "ascend_a2"
            and _is_triton_ascend_321_distribution()
        )
        self._use_raw_pointer_args = os.environ.get(
            "KERNELSWIFT_TASK08_RAW_POINTER_ARGS", "ptr"
        ).strip().lower() in ("ptr", "pointer", "1", "on", "true")
        self._compiled_contract_is_official = (
            self.hc_mult == 4
            and self.sinkhorn_iters == 20
            and self.eps == 1e-6
        )
        self._compiled_runner = None
        self._compiled_runner_key = None

    @staticmethod
    def _tensor_signature(tensor):
        # Triton's pointer specialization distinguishes 16-byte alignment.
        # Keep exact shape/stride/dtype/device guards because the frozen runner
        # intentionally bypasses JITFunction's binder and specialization key.
        return (
            tensor.device,
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.data_ptr() % 16 == 0,
        )

    def _forward_compiled(self, mixes, hc_scale, hc_base):
        tensors = (mixes, hc_scale, hc_base)
        if any(t.device.type != "npu" for t in tensors):
            raise ValueError("Task08's frozen launcher requires NPU tensors")
        if not all(t.device == mixes.device for t in tensors):
            raise ValueError("all inputs must be on the same accelerator")

        b, s, mix_hc = mixes.shape
        expected = (2 + self.hc_mult) * self.hc_mult
        if mix_hc != expected:
            raise ValueError(f"expected mix dim {expected}, got {mix_hc}")

        num_rows = b * s
        if self._pack_outputs:
            pre_elements = num_rows * self.hc_mult
            comb_elements = num_rows * self.hc_mult * self.hc_mult
            post_offset = (
                triton.cdiv(pre_elements, self._packed_alignment)
                * self._packed_alignment
            )
            comb_offset = post_offset + post_offset
            storage = torch.empty(
                comb_offset + comb_elements,
                device=mixes.device,
                dtype=torch.float32,
            )
            pre = storage[:pre_elements].view(b, s, self.hc_mult)
            post = storage[post_offset : post_offset + pre_elements].view(
                b, s, self.hc_mult
            )
            comb = storage[comb_offset : comb_offset + comb_elements].view(
                b, s, self.hc_mult, self.hc_mult
            )
        else:
            pre = torch.empty(
                (b, s, self.hc_mult), device=mixes.device, dtype=torch.float32
            )
            post = torch.empty_like(pre)
            comb = torch.empty(
                (b, s, self.hc_mult, self.hc_mult),
                device=mixes.device,
                dtype=torch.float32,
            )
        if self._rows_per_program == 1:
            kernel = _hc_split_sinkhorn_kernel
            grid3 = (num_rows, 1, 1)
            runtime_args = (mixes, hc_scale, hc_base, pre, post, comb)
            constexpr_key = (
                self.hc_mult,
                self.sinkhorn_iters,
                self.eps,
                self._use_factorized_sinkhorn,
                self._unroll_sinkhorn,
                self._num_warps,
                grid3,
            )
            compile_kwargs = dict(
                hc=self.hc_mult,
                sinkhorn_iters=self.sinkhorn_iters,
                eps=self.eps,
                FACTORIZED_SINKHORN=self._use_factorized_sinkhorn,
                UNROLL_SINKHORN=self._unroll_sinkhorn,
                num_warps=self._num_warps,
            )
        else:
            kernel = _hc_split_sinkhorn_rows_kernel
            grid3 = (triton.cdiv(num_rows, self._rows_per_program), 1, 1)
            runtime_args = (mixes, hc_scale, hc_base, pre, post, comb, num_rows)
            constexpr_key = (
                self.hc_mult,
                self.sinkhorn_iters,
                self.eps,
                self._use_factorized_sinkhorn,
                self._rows_per_program,
                self._unroll_sinkhorn,
                self._num_warps,
                grid3,
            )
            compile_kwargs = dict(
                hc=self.hc_mult,
                sinkhorn_iters=self.sinkhorn_iters,
                eps=self.eps,
                FACTORIZED_SINKHORN=self._use_factorized_sinkhorn,
                ROWS_PER_PROGRAM=self._rows_per_program,
                UNROLL_SINKHORN=self._unroll_sinkhorn,
                num_warps=self._num_warps,
            )

        runner_key = (
            tuple(self._tensor_signature(t) for t in runtime_args if torch.is_tensor(t)),
            constexpr_key,
        )
        if self._compiled_runner is None or self._compiled_runner_key != runner_key:
            compiled = kernel.warmup(
                *runtime_args,
                grid=grid3,
                **compile_kwargs,
            )
            runner = _compiled_grid_runner(compiled, grid3)
            if runner is None:
                return hc_split_sinkhorn(
                    mixes,
                    hc_scale,
                    hc_base,
                    self.hc_mult,
                    self.sinkhorn_iters,
                    self.eps,
                    self._num_warps,
                    self._use_factorized_sinkhorn,
                    self._rows_per_program,
                    self._unroll_sinkhorn,
                    self._pack_outputs,
                    self._packed_alignment,
                )
            self._compiled_runner = runner
            self._compiled_runner_key = runner_key

        # Only non-constexpr arguments are part of the Ascend 3.2.1 launcher
        # ABI.  stream is deliberately omitted so the runner obtains the
        # current NPU stream for every real launch.
        if self._use_raw_pointer_args:
            pointer_args = (
                int(mixes.data_ptr()),
                int(hc_scale.data_ptr()),
                int(hc_base.data_ptr()),
                int(pre.data_ptr()),
                int(post.data_ptr()),
                int(comb.data_ptr()),
            )
            if self._rows_per_program == 1:
                self._compiled_runner(*pointer_args)
            else:
                self._compiled_runner(*pointer_args, num_rows)
        else:
            self._compiled_runner(*runtime_args)
        return pre, post, comb

    def forward(self, mixes, hc_scale, hc_base):
        if (
            self._use_compiled_launcher
            and self._compiled_contract_is_official
            and _matches_frozen_sinkhorn_inputs((mixes, hc_scale, hc_base))
        ):
            return self._forward_compiled(mixes, hc_scale, hc_base)
        return hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.eps,
            self._num_warps,
            self._use_factorized_sinkhorn,
            self._rows_per_program,
            self._unroll_sinkhorn,
            self._pack_outputs,
            self._packed_alignment,
        )


class Model(ModelNew):
    """Compatibility name required by the written competition rules."""


def get_init_inputs():
    return [4, 20, 1e-6]


def get_inputs():
    hc = 4
    mix_hc = (2 + hc) * hc
    torch.manual_seed(0)
    mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32)
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32)
    hc_base = torch.randn(mix_hc, dtype=torch.float32) * 0.1
    return [mixes, hc_scale, hc_base]
