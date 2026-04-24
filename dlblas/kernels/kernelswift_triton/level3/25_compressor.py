import math
import torch_npu  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _num_warps(block_size: int) -> int:
    if block_size <= 64:
        return 1
    if block_size <= 128:
        return 2
    if block_size <= 256:
        return 4
    return 8


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


if True:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        n_rows,
        n_cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        row_start = pid * n_cols
        idx = row_start + offs
        mask = offs < n_cols
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        mean = tl.sum(x_f32 * x_f32, axis=0) * (1.0 / n_cols)
        inv_rms = tl.math.rsqrt(mean + eps)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x_f32 * inv_rms * w
        tl.store(out_ptr + idx, y, mask=mask)


    @triton.jit
    def _overlap_pack_kernel(
        kv_in_ptr,
        score_in_ptr,
        kv_out_ptr,
        score_out_ptr,
        batch_size,
        num_windows,
        ratio,
        head_dim,
        score_fill,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows_per_batch = num_windows * (2 * ratio)
        b = pid // rows_per_batch
        row_in_batch = pid % rows_per_batch
        w = row_in_batch // (2 * ratio)
        rr = row_in_batch % (2 * ratio)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        use_curr = rr >= ratio
        prev_w = tl.maximum(w - 1, 0)
        src_w = tl.where(use_curr, w, prev_w)
        src_r = tl.where(use_curr, rr - ratio, rr)
        src_d_base = tl.where(use_curr, head_dim, 0)
        valid_src = use_curr | (w > 0)

        in_row_offset = (((b * num_windows + src_w) * ratio + src_r) * (2 * head_dim)) + src_d_base
        out_row_offset = (((b * num_windows + w) * (2 * ratio) + rr) * head_dim)

        kv_vals = tl.load(kv_in_ptr + in_row_offset + offs_d, mask=valid_src & mask_d, other=0.0)
        score_vals = tl.load(score_in_ptr + in_row_offset + offs_d, mask=valid_src & mask_d, other=score_fill)

        tl.store(kv_out_ptr + out_row_offset + offs_d, kv_vals, mask=mask_d)
        tl.store(score_out_ptr + out_row_offset + offs_d, score_vals, mask=mask_d)


    @triton.jit
    def _prefill_reduce_kernel(
        kv_ptr,
        score_ptr,
        out_ptr,
        num_rows,
        ratio,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_rows:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        base_offset = pid * ratio * head_dim
        max_vec = tl.full([BLOCK_D], -float("inf"), dtype=tl.float32)
        sum_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for r in range(0, ratio):
            row_offset = base_offset + r * head_dim + offs_d
            score_vec = tl.load(score_ptr + row_offset, mask=mask_d, other=-float("inf")).to(tl.float32)
            kv_vec = tl.load(kv_ptr + row_offset, mask=mask_d, other=0.0).to(tl.float32)

            new_max = tl.maximum(max_vec, score_vec)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = score_vec == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, score_vec) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))

            acc_vec = acc_vec * alpha + kv_vec * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max

        valid_sum = sum_vec > 0.0
        safe_sum = tl.where(valid_sum, sum_vec, 1.0)
        out_vec = tl.where(valid_sum, acc_vec / safe_sum, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out_vec, mask=mask_d)


    @triton.jit
    def _prefill_reduce_overlap_kernel(
        kv_ptr,
        score_ptr,
        out_ptr,
        num_windows,
        ratio,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // num_windows
        w = pid % num_windows

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        max_vec = tl.full([BLOCK_D], -float("inf"), dtype=tl.float32)
        sum_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for rr in range(0, 2 * ratio):
            use_curr = rr >= ratio
            prev_w = tl.maximum(w - 1, 0)
            src_w = tl.where(use_curr, w, prev_w)
            src_r = tl.where(use_curr, rr - ratio, rr)
            src_c = tl.where(use_curr, head_dim + offs_d, offs_d)
            valid_src = use_curr | (w > 0)

            row_offset = (((b * num_windows + src_w) * ratio + src_r) * (2 * head_dim)) + src_c
            score_vec = tl.load(score_ptr + row_offset, mask=valid_src & mask_d, other=-float("inf")).to(tl.float32)
            kv_vec = tl.load(kv_ptr + row_offset, mask=valid_src & mask_d, other=0.0).to(tl.float32)

            new_max = tl.maximum(max_vec, score_vec)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = score_vec == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, score_vec) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))

            acc_vec = acc_vec * alpha + kv_vec * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max

        valid_sum = sum_vec > 0.0
        safe_sum = tl.where(valid_sum, sum_vec, 1.0)
        out_vec = tl.where(valid_sum, acc_vec / safe_sum, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out_vec, mask=mask_d)


    @triton.jit
    def _decode_update_state_kernel(
        kv_token_ptr,
        score_token_ptr,
        ape_ptr,
        kv_state_ptr,
        score_state_ptr,
        state_rows,
        state_slot,
        state_width,
        BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_W)
        mask = offs < state_width

        token_offset = pid * state_width + offs
        state_offset = ((pid * state_rows) + state_slot) * state_width + offs
        kv_vals = tl.load(kv_token_ptr + token_offset, mask=mask, other=0.0)
        score_vals = tl.load(score_token_ptr + token_offset, mask=mask, other=0.0)
        ape_vals = tl.load(ape_ptr + offs, mask=mask, other=0.0)

        tl.store(kv_state_ptr + state_offset, kv_vals, mask=mask)
        tl.store(score_state_ptr + state_offset, score_vals + ape_vals, mask=mask)


    @triton.jit
    def _decode_reduce_nonoverlap_kernel(
        kv_ptr,
        score_ptr,
        out_ptr,
        ratio,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        base_offset = pid * ratio * head_dim
        max_vec = tl.full([BLOCK_D], -float("inf"), dtype=tl.float32)
        sum_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for r in range(0, ratio):
            row_offset = base_offset + r * head_dim + offs_d
            score_vec = tl.load(score_ptr + row_offset, mask=mask_d, other=-float("inf")).to(tl.float32)
            kv_vec = tl.load(kv_ptr + row_offset, mask=mask_d, other=0.0).to(tl.float32)
            new_max = tl.maximum(max_vec, score_vec)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = score_vec == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, score_vec) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))
            acc_vec = acc_vec * alpha + kv_vec * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max

        valid_sum = sum_vec > 0.0
        safe_sum = tl.where(valid_sum, sum_vec, 1.0)
        out_vec = tl.where(valid_sum, acc_vec / safe_sum, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out_vec, mask=mask_d)


    @triton.jit
    def _decode_update_reduce_nonoverlap_kernel(
        kv_token_ptr,
        score_token_ptr,
        ape_ptr,
        kv_state_ptr,
        score_state_ptr,
        out_ptr,
        ratio,
        state_slot,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        token_offset = pid * head_dim + offs_d
        state_base = pid * ratio * head_dim
        slot_offset = state_base + state_slot * head_dim + offs_d

        kv_new = tl.load(kv_token_ptr + token_offset, mask=mask_d, other=0.0).to(tl.float32)
        score_new = (
            tl.load(score_token_ptr + token_offset, mask=mask_d, other=0.0).to(tl.float32)
            + tl.load(ape_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        )
        tl.store(kv_state_ptr + slot_offset, kv_new, mask=mask_d)
        tl.store(score_state_ptr + slot_offset, score_new, mask=mask_d)

        max_vec = tl.full([BLOCK_D], -float("inf"), dtype=tl.float32)
        sum_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for r in range(0, ratio):
            row_offset = state_base + r * head_dim + offs_d
            kv_vec = tl.load(kv_state_ptr + row_offset, mask=mask_d, other=0.0).to(tl.float32)
            score_vec = tl.load(score_state_ptr + row_offset, mask=mask_d, other=-float("inf")).to(tl.float32)
            new_max = tl.maximum(max_vec, score_vec)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = score_vec == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, score_vec) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))
            acc_vec = acc_vec * alpha + kv_vec * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max

        valid_sum = sum_vec > 0.0
        safe_sum = tl.where(valid_sum, sum_vec, 1.0)
        out_vec = tl.where(valid_sum, acc_vec / safe_sum, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out_vec, mask=mask_d)


    @triton.jit
    def _decode_reduce_overlap_kernel(
        kv_ptr,
        score_ptr,
        out_ptr,
        ratio,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        state_rows = 2 * ratio
        state_width = 2 * head_dim
        base_offset = pid * state_rows * state_width
        max_vec = tl.full([BLOCK_D], -float("inf"), dtype=tl.float32)
        sum_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for r in range(0, 2 * ratio):
            src_row = tl.where(r < ratio, r, ratio + (r - ratio))
            src_col = tl.where(r < ratio, offs_d, head_dim + offs_d)
            row_offset = base_offset + src_row * state_width + src_col
            score_vec = tl.load(score_ptr + row_offset, mask=mask_d, other=-float("inf")).to(tl.float32)
            kv_vec = tl.load(kv_ptr + row_offset, mask=mask_d, other=0.0).to(tl.float32)
            new_max = tl.maximum(max_vec, score_vec)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = score_vec == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, score_vec) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))
            acc_vec = acc_vec * alpha + kv_vec * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max

        valid_sum = sum_vec > 0.0
        safe_sum = tl.where(valid_sum, sum_vec, 1.0)
        out_vec = tl.where(valid_sum, acc_vec / safe_sum, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out_vec, mask=mask_d)


    @triton.jit
    def _decode_update_reduce_overlap_kernel(
        kv_token_ptr,
        score_token_ptr,
        ape_ptr,
        kv_state_ptr,
        score_state_ptr,
        out_ptr,
        ratio,
        state_slot,
        head_dim,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim

        state_rows = 2 * ratio
        state_width = 2 * head_dim
        state_base = pid * state_rows * state_width
        token_base = pid * state_width
        slot_row_base = state_base + (ratio + state_slot) * state_width

        kv_lo = tl.load(kv_token_ptr + token_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        kv_hi = tl.load(kv_token_ptr + token_base + head_dim + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        score_lo = (
            tl.load(score_token_ptr + token_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            + tl.load(ape_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        )
        score_hi = (
            tl.load(score_token_ptr + token_base + head_dim + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            + tl.load(ape_ptr + head_dim + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        )

        tl.store(kv_state_ptr + slot_row_base + offs_d, kv_lo, mask=mask_d)
        tl.store(kv_state_ptr + slot_row_base + head_dim + offs_d, kv_hi, mask=mask_d)
        tl.store(score_state_ptr + slot_row_base + offs_d, score_lo, mask=mask_d)
        tl.store(score_state_ptr + slot_row_base + head_dim + offs_d, score_hi, mask=mask_d)

        max_vec = tl.full([BLOCK_D], -float("inf"), dtype=tl.float32)
        sum_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_vec = tl.zeros([BLOCK_D], dtype=tl.float32)

        for r in range(0, ratio):
            prev_row = state_base + r * state_width + offs_d
            curr_row = state_base + (ratio + r) * state_width + head_dim + offs_d

            prev_score = tl.load(score_state_ptr + prev_row, mask=mask_d, other=-float("inf")).to(tl.float32)
            prev_kv = tl.load(kv_state_ptr + prev_row, mask=mask_d, other=0.0).to(tl.float32)
            new_max = tl.maximum(max_vec, prev_score)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = prev_score == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, prev_score) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))
            acc_vec = acc_vec * alpha + prev_kv * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max

            curr_score = tl.load(score_state_ptr + curr_row, mask=mask_d, other=-float("inf")).to(tl.float32)
            curr_kv = tl.load(kv_state_ptr + curr_row, mask=mask_d, other=0.0).to(tl.float32)
            new_max = tl.maximum(max_vec, curr_score)
            prev_is_inf = max_vec == -float("inf")
            score_is_inf = curr_score == -float("inf")
            prev_delta = tl.where(prev_is_inf, 0.0, max_vec) - tl.where(prev_is_inf, 0.0, new_max)
            score_delta = tl.where(score_is_inf, 0.0, curr_score) - tl.where(score_is_inf, 0.0, new_max)
            alpha = tl.where(prev_is_inf, 0.0, tl.exp(prev_delta))
            beta = tl.where(score_is_inf, 0.0, tl.exp(score_delta))
            acc_vec = acc_vec * alpha + curr_kv * beta
            sum_vec = sum_vec * alpha + beta
            max_vec = new_max
        valid_sum = sum_vec > 0.0
        safe_sum = tl.where(valid_sum, sum_vec, 1.0)
        out_vec = tl.where(valid_sum, acc_vec / safe_sum, 0.0)
        tl.store(out_ptr + pid * head_dim + offs_d, out_vec, mask=mask_d)

        for r in range(0, ratio):
            src_row = state_base + (ratio + r) * state_width
            dst_row = state_base + r * state_width
            src_lo = tl.load(kv_state_ptr + src_row + offs_d, mask=mask_d, other=0.0)
            src_hi = tl.load(kv_state_ptr + src_row + head_dim + offs_d, mask=mask_d, other=0.0)
            src_score_lo = tl.load(score_state_ptr + src_row + offs_d, mask=mask_d, other=-float("inf"))
            src_score_hi = tl.load(score_state_ptr + src_row + head_dim + offs_d, mask=mask_d, other=-float("inf"))
            tl.store(kv_state_ptr + dst_row + offs_d, src_lo, mask=mask_d)
            tl.store(kv_state_ptr + dst_row + head_dim + offs_d, src_hi, mask=mask_d)
            tl.store(score_state_ptr + dst_row + offs_d, src_score_lo, mask=mask_d)
            tl.store(score_state_ptr + dst_row + head_dim + offs_d, src_score_hi, mask=mask_d)


    @triton.jit
    def _overlap_roll_state_kernel(
        kv_state_ptr,
        score_state_ptr,
        ratio,
        state_rows,
        state_width,
        BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_per_row = tl.cdiv(state_width, BLOCK_W)
        batch = pid // (ratio * blocks_per_row)
        row = (pid // blocks_per_row) % ratio
        block = pid % blocks_per_row
        offs = block * BLOCK_W + tl.arange(0, BLOCK_W)
        mask = offs < state_width
        batch_base = batch * state_rows * state_width

        src_offset = batch_base + (ratio + row) * state_width + offs
        dst_offset = batch_base + row * state_width + offs

        kv_vals = tl.load(kv_state_ptr + src_offset, mask=mask, other=0.0)
        score_vals = tl.load(score_state_ptr + src_offset, mask=mask, other=0.0)
        tl.store(kv_state_ptr + dst_offset, kv_vals, mask=mask)
        tl.store(score_state_ptr + dst_offset, score_vals, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        n_cols = x.shape[-1]
        x_flat = x.contiguous().view(-1, n_cols)
        n_rows = x_flat.shape[0]
        out_flat = torch.empty_like(x_flat)
        block = 1 << (int(n_cols - 1).bit_length())
        block = max(64, min(1024, block))
        _rmsnorm_fwd_kernel[(n_rows,)](
            x_flat,
            self.weight,
            out_flat,
            n_rows,
            n_cols,
            self.eps,
            BLOCK=block,
            num_warps=_num_warps(block),
            num_stages=1,
        )
        return out_flat.view_as(x).to(orig_dtype)


def precompute_freqs_cis(dim: int, seqlen: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_complex = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    y.copy_(x_complex)
    return y


def overlap_pack(
    kv_windows: torch.Tensor,
    score_windows: torch.Tensor,
    head_dim: int,
    fill_score_neg_inf: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_windows, ratio, width = kv_windows.shape
    if width != 2 * head_dim or score_windows.shape != kv_windows.shape:
        raise ValueError("overlap_pack expects [..., ratio, 2 * head_dim] tensors")
    kv_in = kv_windows.contiguous()
    score_in = score_windows.contiguous()
    packed_kv = torch.empty((batch_size, num_windows, 2 * ratio, head_dim), device=kv_in.device, dtype=kv_in.dtype)
    packed_score = torch.empty(
        (batch_size, num_windows, 2 * ratio, head_dim),
        device=score_in.device,
        dtype=score_in.dtype,
    )
    grid = (batch_size * num_windows * (2 * ratio),)
    block_d = min(256, _next_power_of_2(head_dim))
    score_fill = float("-inf") if fill_score_neg_inf else 0.0
    _overlap_pack_kernel[grid](
        kv_in,
        score_in,
        packed_kv,
        packed_score,
        batch_size,
        num_windows,
        ratio,
        head_dim,
        score_fill,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
        num_stages=1,
    )
    return packed_kv, packed_score


def prefill_reduce(kv_windows: torch.Tensor, score_windows: torch.Tensor) -> torch.Tensor:
    kv_in = kv_windows.contiguous()
    score_in = score_windows.contiguous()
    batch_size, num_windows, ratio, head_dim = kv_in.shape
    out = torch.empty((batch_size, num_windows, head_dim), device=kv_in.device, dtype=torch.float32)
    grid = (batch_size * num_windows,)
    block_d = min(256, _next_power_of_2(head_dim))
    _prefill_reduce_kernel[grid](
        kv_in,
        score_in,
        out,
        batch_size * num_windows,
        ratio,
        head_dim,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
        num_stages=1,
    )
    return out.to(kv_in.dtype)


def prefill_reduce_overlap(kv_windows: torch.Tensor, score_windows: torch.Tensor, head_dim: int) -> torch.Tensor:
    kv_in = kv_windows.contiguous()
    score_in = score_windows.contiguous()
    batch_size, num_windows, ratio, width = kv_in.shape
    if width != 2 * head_dim:
        raise ValueError("prefill_reduce_overlap expects [..., ratio, 2 * head_dim] tensors")
    out = torch.empty((batch_size, num_windows, head_dim), device=kv_in.device, dtype=torch.float32)
    block_d = min(256, _next_power_of_2(head_dim))
    _prefill_reduce_overlap_kernel[(batch_size * num_windows,)](
        kv_in,
        score_in,
        out,
        num_windows,
        ratio,
        head_dim,
        BLOCK_D=block_d,
        num_warps=_num_warps(block_d),
        num_stages=1,
    )
    return out.to(kv_in.dtype)


def decode_update_state(
    kv_token: torch.Tensor,
    score_token: torch.Tensor,
    ape_row: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    slot: int,
    overlap: bool,
) -> None:
    ratio = kv_state.size(1) // (1 + int(overlap))
    state_slot = ratio + slot if overlap else slot
    kv_in = kv_token.squeeze(1).contiguous()
    score_in = score_token.squeeze(1).contiguous()
    ape_in = ape_row.contiguous()
    state_width = kv_state.size(-1)
    block_w = min(256, _next_power_of_2(state_width))
    grid = (kv_state.size(0),)
    _decode_update_state_kernel[grid](
        kv_in,
        score_in,
        ape_in,
        kv_state,
        score_state,
        kv_state.size(1),
        state_slot,
        state_width,
        BLOCK_W=block_w,
        num_warps=_num_warps(block_w),
        num_stages=1,
    )


def decode_update_reduce(
    kv_token: torch.Tensor,
    score_token: torch.Tensor,
    ape_row: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ratio: int,
    head_dim: int,
    slot: int,
    overlap: bool,
) -> torch.Tensor:
    out = torch.empty((kv_state.size(0), head_dim), device=kv_state.device, dtype=torch.float32)
    block_d = min(256, _next_power_of_2(head_dim))
    if overlap:
        _decode_update_reduce_overlap_kernel[(kv_state.size(0),)](
            kv_token.squeeze(1).contiguous(),
            score_token.squeeze(1).contiguous(),
            ape_row.contiguous(),
            kv_state,
            score_state,
            out,
            ratio,
            slot,
            head_dim,
            BLOCK_D=block_d,
            num_warps=_num_warps(block_d),
            num_stages=1,
        )
    else:
        _decode_update_reduce_nonoverlap_kernel[(kv_state.size(0),)](
            kv_token.squeeze(1).contiguous(),
            score_token.squeeze(1).contiguous(),
            ape_row.contiguous(),
            kv_state,
            score_state,
            out,
            ratio,
            slot,
            head_dim,
            BLOCK_D=block_d,
            num_warps=_num_warps(block_d),
            num_stages=1,
        )
    return out.unsqueeze(1).to(kv_state.dtype)


def decode_reduce(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ratio: int,
    head_dim: int,
    overlap: bool,
) -> torch.Tensor:
    out = torch.empty((kv_state.size(0), head_dim), device=kv_state.device, dtype=torch.float32)
    block_d = min(256, _next_power_of_2(head_dim))
    if overlap:
        _decode_reduce_overlap_kernel[(kv_state.size(0),)](
            kv_state.contiguous(),
            score_state.contiguous(),
            out,
            ratio,
            head_dim,
            BLOCK_D=block_d,
            num_warps=_num_warps(block_d),
            num_stages=1,
        )
    else:
        _decode_reduce_nonoverlap_kernel[(kv_state.size(0),)](
            kv_state.contiguous(),
            score_state.contiguous(),
            out,
            ratio,
            head_dim,
            BLOCK_D=block_d,
            num_warps=_num_warps(block_d),
            num_stages=1,
        )
    return out.unsqueeze(1).to(kv_state.dtype)


class ModelNew(nn.Module):
    def __init__(
        self,
        max_batch_size: int = 4,
        max_seq_len: int = 256,
        dim: int = 512,
        head_dim: int = 128,
        rope_head_dim: int = 64,
        compress_ratio: int = 4,
        norm_eps: float = 1e-6,
    ):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coeff = 1 + int(self.overlap)

        self.ape = nn.Parameter(torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32))
        self.wkv = Linear(dim, coeff * head_dim, dtype=torch.float32)
        self.wgate = Linear(dim, coeff * head_dim, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, norm_eps)

        self.register_buffer(
            "kv_state",
            torch.zeros(max_batch_size, coeff * compress_ratio, coeff * head_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (max_batch_size, coeff * compress_ratio, coeff * head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, max_seq_len // compress_ratio, head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(rope_head_dim, max_seq_len),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ape, mean=0.0, std=0.02)
        self.wkv.reset_parameters()
        self.wgate.reset_parameters()
        nn.init.ones_(self.norm.weight)

    def reset_runtime_state(self) -> None:
        self.kv_state.zero_()
        self.score_state.fill_(float("-inf"))
        self.kv_cache.zero_()

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        batch_size, seqlen, _ = x.shape
        ratio = self.compress_ratio
        head_dim = self.head_dim
        rope_head_dim = self.rope_head_dim
        overlap = self.overlap
        dtype = x.dtype

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            if overlap and cutoff >= ratio:
                self.kv_state[:batch_size, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:batch_size, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape

            if remainder > 0:
                self.kv_state[:batch_size, offset : offset + remainder] = kv[:, cutoff:]
                self.score_state[:batch_size, offset : offset + remainder] = score[:, cutoff:] + self.ape[:remainder]

            if not should_compress:
                return None

            kv_windows = kv[:, :cutoff].unflatten(1, (-1, ratio))
            score_windows = score[:, :cutoff].unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = prefill_reduce_overlap(kv_windows, score_windows, head_dim)
            else:
                kv = prefill_reduce(kv_windows, score_windows)
        else:
            slot = start_pos % ratio
            should_compress = (start_pos + 1) % ratio == 0
            if not should_compress:
                decode_update_state(
                    kv_token=kv,
                    score_token=score,
                    ape_row=self.ape[slot],
                    kv_state=self.kv_state[:batch_size],
                    score_state=self.score_state[:batch_size],
                    slot=slot,
                    overlap=overlap,
                )
                return None
            kv = decode_update_reduce(
                kv_token=kv,
                score_token=score,
                ape_row=self.ape[slot],
                kv_state=self.kv_state[:batch_size],
                score_state=self.score_state[:batch_size],
                ratio=ratio,
                head_dim=head_dim,
                slot=slot,
                overlap=overlap,
            )

        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio].to(kv.device)
            self.kv_cache[:batch_size, : seqlen // ratio] = kv
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0).to(kv.device)
            self.kv_cache[:batch_size, start_pos // ratio] = kv.squeeze(1)
        apply_rotary_emb(kv[..., -rope_head_dim:], freqs_cis)
        return kv


def generate_test_data(params: dict) -> tuple[torch.Tensor, int]:
    batch_size = params["batch_size"]
    seq_len = params["seq_len"]
    dim = params["dim"]
    start_pos = params["start_pos"]
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device="cpu")
    return x, start_pos


def test_kv_compress():
    return ModelNew(*get_init_inputs()).forward(*get_inputs())


def get_inputs():
    params = {"batch_size": 1, "seq_len": 12, "dim": 448, "start_pos": 0}
    return list(generate_test_data(params))


def get_init_inputs():
    return [1, 256, 448, 32, 4, 4, 1e-6]
