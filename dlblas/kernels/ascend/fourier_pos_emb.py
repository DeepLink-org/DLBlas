from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from dlblas.utils.device_utils import NUM_CORES, DEVICE
from dlblas.utils.op_helper import grouped_launch_diagonal


class ModelConfig:
    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: Optional[int] = None
    fourier_dim: int = 0
    fourier_ignore_zero: bool = True
    fourier_separate_head: bool = True
    fourier_separate_basis: bool = True
    fourier_norm: bool = False


class FourierEmbedding(nn.Module):
    def __init__(self, config, *args, **kwargs):
        self.config: ModelConfig = config

        self.head_dim = self.config.d_model // self.config.n_heads
        dim = (
            self.config.fourier_dim
            if self.config.fourier_dim > self.head_dim
            else self.head_dim
        )

        super().__init__(config, dim=dim, *args, **kwargs)

        self.suffix = "fourier"

        if self.config.fourier_ignore_zero:
            self.input_dim = self.inv_freq.size(-1)
            self.output_dim = min(
                self.input_dim, self.head_dim // 4
            )  # TODO: self.head_dim//8
        else:
            self.input_dim = self.dim // 2
            self.output_dim = self.head_dim // 2

        if self.prefix == "embed":
            self.input_shape = "btD"
            self.output_shape = "btd"
        elif self.prefix == "attn":
            self.input_shape = "bhtD"
            self.output_shape = "bhtd"

        if self.prefix == "attn" and self.config.fourier_separate_head:
            size = (self.config.n_heads, self.input_dim, self.output_dim)
            self.coef_shape = "hDd"
        else:
            size = (self.input_dim, self.output_dim)
            self.coef_shape = "Dd"

        if self.config.fourier_separate_basis:
            self.sin_coef = torch.randn(size=size, device=DEVICE, dtype=torch.float)
            self.cos_coef = torch.randn(size=size, device=DEVICE, dtype=torch.float)
        else:
            self.fourier_coef = torch.randn(size=size, device=DEVICE, dtype=torch.float)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        if self.prefix == "embed":
            B, T, hs = x.size()
            x = x.view(B, T, 2, hs // 2)

        elif self.prefix == "attn":
            B, nh, T, hs = x.size()
            x = x.view(B, nh, T, 2, hs // 2)

        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t, inverse=False):
        if self.config.fourier_separate_basis:
            if self.config.fourier_norm:
                fourier_sin = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_sin,
                    self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True),
                )
                fourier_cos = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_cos,
                    self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True),
                )
            else:
                fourier_sin = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_sin,
                    self.sin_coef,
                )
                fourier_cos = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_cos,
                    self.cos_coef,
                )
        else:
            if self.config.fourier_norm:
                fourier_sin = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_sin,
                    self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True),
                )
                fourier_cos = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_cos,
                    self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True),
                )
            else:
                fourier_sin = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_sin,
                    self.fourier_coef,
                )
                fourier_cos = torch.einsum(
                    f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}",
                    pos_cos,
                    self.fourier_coef,
                )

        if self.config.fourier_ignore_zero:
            fourier_sin = F.pad(
                input=fourier_sin,
                pad=(0, self.head_dim // 2 - fourier_sin.size(-1)),
                mode="constant",
                value=1,
            )
            fourier_cos = F.pad(
                input=fourier_cos,
                pad=(0, self.head_dim // 2 - fourier_cos.size(-1)),
                mode="constant",
                value=1,
            )

        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)

        if not inverse:
            return ((t * fourier_cos) + (self.rotate_half(t) * fourier_sin)).to(t.dtype)
        else:
            return ((t * fourier_cos) - (self.rotate_half(t) * fourier_sin)).to(t.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, 2, hs // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def fourier_rope_0_ref(
    pos_sin: torch.Tensor,
    pos_cos: torch.Tensor,
    sin_coef: torch.Tensor,
    cos_coef: torch.Tensor,
    t: torch.Tensor,
    inverse=False,
    prefix="attn",
    fourier_separate_basis=True,
    fourier_norm=False,
    fourier_ignore_zero=True,
):
    """
    Args:
        pos_sin (torch.Tensor): Input tensor of shape (b, h, t, D).
        pos_cos (torch.Tensor): Input tensor of shape (b, h, t, D).
        sin_coef (torch.Tensor): Input tensor of shape (h, D, d).
        cos_coef (torch.Tensor): Input tensor of shape (h, D, d).
        t (torch.Tensor): Input tensor of shape (b, h, t, d).
    """

    # d_model = 768
    # n_heads = 12
    # head_dim = d_model // n_heads
    # assert t.shape[-1] == head_dim
    fourier_sin = torch.einsum("bhtD, hDd -> bhtd", pos_sin, sin_coef)
    fourier_cos = torch.einsum("bhtD, hDd -> bhtd", pos_cos, cos_coef)

    fourier_sin = F.pad(
        input=fourier_sin,
        pad=(0, t.shape[-1] // 2 - fourier_sin.size(-1)),
        mode="constant",
        value=1,
    )
    fourier_cos = F.pad(
        input=fourier_cos,
        pad=(0, t.shape[-1] // 2 - fourier_cos.size(-1)),
        mode="constant",
        value=1,
    )
    fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
    fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
    return ((t * fourier_cos) + (rotate_half(t) * fourier_sin)).to(t.dtype)


@triton.jit
def _single_rope_matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for head_idx in range(HEADS):
        for batch_idx in range(BATCH):
            for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
                task_m_idx, task_n_idx = grouped_launch_diagonal(
                    block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
                )
                m_start = task_m_idx * BLOCK_M
                n_start = task_n_idx * BLOCK_N
                mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_start in range(0, K, BLOCK_K):
                    mat_a_offset = (
                        (batch_idx * HEADS * M * K)
                        + (head_idx * M * K)
                        + ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None]
                        + (k_start + tl.arange(0, BLOCK_K))[None, :]
                    )
                    mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                        (k_start + tl.arange(0, BLOCK_K)) < K
                    )[None, :]
                    mat_a_block = tl.load(
                        mat_a + mat_a_offset, mask=mat_a_mask, other=0.0
                    )
                    dl.compile_hint(mat_a_block, "dot_pad_only_k")
                    mat_b_offset = (
                        (head_idx * K * N)
                        + ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None]
                        + (n_start + tl.arange(0, BLOCK_N))[None, :]
                    )
                    mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                        (n_start + tl.arange(0, BLOCK_N)) < N
                    )[None, :]
                    mat_b_block = tl.load(
                        mat_b + mat_b_offset, mask=mat_b_mask, other=0.0
                    )
                    dl.compile_hint(mat_b_block, "dot_pad_only_k")
                    mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
                mat_c_offset = (
                    (batch_idx * HEADS * M * N)
                    + (head_idx * M * N)
                    + ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None]
                    + (n_start + tl.arange(0, BLOCK_N))[None, :]
                )
                mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                tl.store(
                    mat_c + mat_c_offset,
                    mat_c_block.to(mat_c.dtype.element_ty),
                    mask=mat_c_mask,
                )


@triton.jit
def _rope_matmul_kernel(
    pos_sin,
    pos_cos,
    sin_coef,
    cos_coef,
    fourier_sin,
    fourier_cos,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    _single_rope_matmul_kernel(
        pos_sin,
        sin_coef,
        fourier_sin,
        M,
        N,
        K,
        NUM_CORES,
        BATCH,
        HEADS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_TRESHHOLD,
    )
    _single_rope_matmul_kernel(
        pos_cos,
        cos_coef,
        fourier_cos,
        M,
        N,
        K,
        NUM_CORES,
        BATCH,
        HEADS,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_TRESHHOLD,
    )


@triton.jit
def apply_rotary_pos_emb_kernel(
    X,
    COS,
    SIN,
    O,
    seq_len,
    DIM: tl.constexpr,
    CS_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(seq_len, BLOCK)
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < seq_len
        half_dim: tl.constexpr = DIM // 2
        feat_offset_ll = tl.arange(0, CS_DIM)
        feat_offset_lh = tl.arange(0, half_dim - CS_DIM) + CS_DIM

        feat_offset_hl = half_dim + feat_offset_ll
        feat_offset_hh = half_dim + feat_offset_lh
        seq_mask = pos_mask[:, None]
        cs_range = tl.arange(0, CS_DIM)
        cs_offset_l = pos_offset[:, None] * CS_DIM + cs_range[None, :]
        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        cos_h = cos_l
        sin_h = sin_l
        base_offset = pos_offset[:, None] * DIM
        x_ll = tl.load(X + base_offset + feat_offset_ll[None, :], mask=seq_mask)
        x_lh = tl.load(X + base_offset + feat_offset_lh[None, :], mask=seq_mask)
        x_hl = tl.load(X + base_offset + feat_offset_hl[None, :], mask=seq_mask)
        x_hh = tl.load(X + base_offset + feat_offset_hh[None, :], mask=seq_mask)
        o_ll = x_ll * cos_l - x_hl * sin_l
        o_lh = x_lh - x_hh

        o_hl = x_hl * cos_h + x_ll * sin_h
        o_hh = x_hh + x_lh

        tl.store(O + base_offset + feat_offset_ll[None, :], o_ll, mask=seq_mask)
        tl.store(O + base_offset + feat_offset_lh[None, :], o_lh, mask=seq_mask)
        tl.store(O + base_offset + feat_offset_hl[None, :], o_hl, mask=seq_mask)
        tl.store(O + base_offset + feat_offset_hh[None, :], o_hh, mask=seq_mask)


def apply_rotary_pos_emb_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    assert x.is_contiguous()
    x_embed = torch.empty_like(x)
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == x.numel() // x.size(-1)
    BLOCK = 32
    apply_rotary_pos_emb_kernel[(NUM_CORES,)](
        x,
        cos,
        sin,
        x_embed,
        seq_len=seq_len,
        DIM=x.size(-1),
        CS_DIM=cos.size(-1),
        BLOCK=BLOCK,
        NUM_CORES=NUM_CORES,
    )
    return x_embed


def rope_triton_fused(
    pos_sin: torch.Tensor,  # [B, H, T, D_POS]
    pos_cos: torch.Tensor,  # [B, H, T, D_POS]
    sin_coef: torch.Tensor,  # [H, D_POS, D_FOURIER]
    cos_coef: torch.Tensor,  # [H, D_POS, D_FOURIER]
    t: torch.Tensor,  # [B, H, T, HEAD_DIM]
) -> torch.Tensor:
    """
    Triton 融合实现, 等价于 rope_0_ref 核心计算逻辑。
    要求: HEAD_DIM 为偶数, D_FOURIER <= HEAD_DIM//2
    """
    assert t.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert pos_sin.shape == pos_cos.shape
    assert sin_coef.shape == cos_coef.shape
    assert t.is_contiguous()

    B, H, T, HEAD_DIM = t.shape
    _, _, _, D_POS = pos_sin.shape
    _, D_POS_CHK, D_FOURIER = sin_coef.shape
    assert D_POS == D_POS_CHK and D_FOURIER <= HEAD_DIM // 2
    m, k, n = T, D_POS, D_FOURIER
    fourier_sin = torch.empty((B, H, T, D_FOURIER), dtype=pos_sin.dtype, device=DEVICE)
    fourier_cos = torch.empty((B, H, T, D_FOURIER), dtype=pos_cos.dtype, device=DEVICE)
    _rope_matmul_kernel[(NUM_CORES,)](
        pos_sin,
        pos_cos,
        sin_coef,
        cos_coef,
        fourier_sin,
        fourier_cos,
        m,
        n,
        k,
        NUM_CORES,
        B,
        H,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        BLOCK_TRESHHOLD=8,
    )
    return apply_rotary_pos_emb_triton(t, fourier_cos, fourier_sin)


def test_rope_0():
    B, H, T, D_POS = 10, 12, 2048, 512
    D_FOURIER = 128
    HEAD_DIM = 512
    TYPE_ = torch.float16
    pos_sin = torch.randn((B, H, T, D_POS), dtype=TYPE_, device=DEVICE)
    pos_cos = torch.randn((B, H, T, D_POS), dtype=TYPE_, device=DEVICE)
    sin_coef = torch.randn((H, D_POS, D_FOURIER), dtype=TYPE_, device=DEVICE)
    cos_coef = torch.randn((H, D_POS, D_FOURIER), dtype=TYPE_, device=DEVICE)
    t = torch.randn((B, H, T, HEAD_DIM), dtype=TYPE_, device=DEVICE)
    out_triton = rope_triton_fused(pos_sin, pos_cos, sin_coef, cos_coef, t)
    out_ref = fourier_rope_0_ref(pos_sin, pos_cos, sin_coef, cos_coef, t)
    torch.testing.assert_close(out_ref, out_triton, rtol=1e-02, atol=1e-02)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["cnt"],  # Argument names to use as an x-axis for the plot
            x_vals=[1],  # NOTE: the tunning framework specialized to one shape
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=["triton", "torch"],  # Label name for the lines
            line_names=["Triton", "Torch"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="fourier_rope-performance",
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(cnt, provider):
        warmup = 200
        rep = 200
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fourier_rope_0_ref(pos_sin, pos_cos, sin_coef, cos_coef, t),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rope_triton_fused(pos_sin, pos_cos, sin_coef, cos_coef, t),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)
    print("run test success")


if __name__ == "__main__":
    test_rope_0()
