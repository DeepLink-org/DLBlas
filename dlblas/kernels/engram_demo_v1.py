"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only:
   This code is a demonstration version intended to illustrate the core logic and
   data flow of the Engram module.

2. Production Readiness:
   This implementation requires further optimization for actual production use
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications:
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection
   mechanisms are omitted or mocked in this version to focus exclusively on the
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import math

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.nn.functional as F
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex


engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()
device = "cuda"


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []

        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                input_ids, layer_id=layer_id
            )
        return hash_ids_for_all_layers


class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long, device=device)
        )

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(
            num_embeddings=total_N, embedding_dim=D, device=device
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)

        return output


# ----------------------------------------------------------------------
#  Kernel 1: Gate computation, value projection, and RMS of value
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32, num_stages=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=32, num_stages=1),
    ],
    key=["C"],
)
@triton.jit
def engram_gate_value_rms_kernel(
    key_ptr,
    query_ptr,
    value_proj_ptr,
    w_key_norm_ptr,
    w_query_norm_ptr,
    output_ptr,
    rms_ptr,
    B,
    T,
    G,
    C,
    stride_kb,
    stride_kt,
    stride_kg,
    stride_kc,
    stride_qb,
    stride_qt,
    stride_qg,
    stride_qc,
    stride_vb,
    stride_vt,
    stride_vc,
    stride_wkg,
    stride_wkc,
    stride_wqg,
    stride_wqc,
    stride_ob,
    stride_ot,
    stride_og,
    stride_oc,
    stride_rms_b,
    stride_rms_t,
    stride_rms_g,
    eps,
    sqrt_C,
    norm_eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B * T * G:
        return
    b = pid // (T * G)
    remainder = pid % (T * G)
    t = remainder // G
    g = remainder % G

    key_start = b * stride_kb + t * stride_kt + g * stride_kg
    query_start = b * stride_qb + t * stride_qt + g * stride_qg
    value_proj_start = b * stride_vb + t * stride_vt

    sum_sq_key = 0.0
    sum_sq_query = 0.0
    dot_raw = 0.0

    inv_C = 1.0 / C

    for off in range(0, C, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < C
        # Load as half and convert to float32 for computation
        k = tl.load(key_ptr + key_start + idx * stride_kc, mask=mask).to(tl.float32)
        q = tl.load(query_ptr + query_start + idx * stride_qc, mask=mask).to(tl.float32)
        w_k = tl.load(w_key_norm_ptr + g * stride_wkg + idx * stride_wkc, mask=mask).to(
            tl.float32
        )
        w_q = tl.load(
            w_query_norm_ptr + g * stride_wqg + idx * stride_wqc, mask=mask
        ).to(tl.float32)

        k_sq = k * k
        q_sq = q * q
        sum_sq_key += tl.sum(k_sq, axis=0)
        sum_sq_query += tl.sum(q_sq, axis=0)
        dot_raw += tl.sum(k * w_k * q * w_q, axis=0)

    rms_key = tl.sqrt(sum_sq_key * inv_C + eps)
    rms_query = tl.sqrt(sum_sq_query * inv_C + eps)
    norm_factor = rms_key * rms_query
    dot_norm = dot_raw / norm_factor
    gate_raw = dot_norm * (1.0 / sqrt_C)

    gate_abs = tl.abs(gate_raw)
    gate_clamped = tl.maximum(gate_abs, 1e-6)
    gate_sqrt = tl.sqrt(gate_clamped)
    gate = gate_sqrt * tl.where(gate_raw >= 0, 1.0, -1.0)
    gate = tl.sigmoid(gate)

    sum_sq_val = 0.0
    for off in range(0, C, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < C
        v = tl.load(value_proj_ptr + value_proj_start + idx * stride_vc, mask=mask).to(
            tl.float32
        )
        out = v * gate
        # store as half to reduce memory traffic
        tl.store(
            output_ptr
            + b * stride_ob
            + t * stride_ot
            + g * stride_og
            + idx * stride_oc,
            out,
            mask=mask,
        )
        out_sq = out * out
        sum_sq_val += tl.sum(out_sq, axis=0)

    rms_val = tl.sqrt(sum_sq_val * inv_C + norm_eps)
    tl.store(rms_ptr + b * stride_rms_b + t * stride_rms_t + g * stride_rms_g, rms_val)


# ----------------------------------------------------------------------
#  Kernel 2: Fused depthwise convolution, activation, and residual addition
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_C": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_SIZE_C": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_C": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE_C": 512}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=32, num_stages=4),
        triton.Config({"BLOCK_SIZE_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_C": 512}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE_C": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_C": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE_C": 512}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE_C": 1024}, num_warps=32, num_stages=1),
    ],
    key=["C"],
)
@triton.jit
def engram_conv_residual_kernel(
    value_ptr,
    rms_ptr,
    w_norm_ptr,
    conv_weight_ptr,
    output_ptr,
    B,
    T,
    G,
    C,
    K,
    dilation,
    stride_vb,
    stride_vt,
    stride_vg,
    stride_vc,
    stride_rms_b,
    stride_rms_t,
    stride_rms_g,
    stride_wn_g,
    stride_wn_c,
    stride_w_ch,
    stride_w_k,
    stride_ob,
    stride_ot,
    stride_og,
    stride_oc,
    activation: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)

    # Number of chunks in C dimension
    num_chunks = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    total_blocks = B * T * G * num_chunks
    if pid >= total_blocks:
        return

    # Decompose pid into btg_id and chunk index
    blocks_per_btg = num_chunks
    btg_id = pid // blocks_per_btg
    chunk = pid % blocks_per_btg

    # Decompose btg_id into b, t, g
    b = btg_id // (T * G)
    rest = btg_id % (T * G)
    t = rest // G
    g = rest % G

    # Channel indices for this block
    c_idx = chunk * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask = c_idx < C

    # Load normalization weights as half and convert to float32
    w_norm_vals = tl.load(
        w_norm_ptr + g * stride_wn_g + c_idx * stride_wn_c, mask=mask, other=0.0
    ).to(tl.float32)

    # Accumulator for convolution
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)

    # Loop over kernel taps
    for k in range(K):
        t_in = t - (K - 1 - k) * dilation
        if t_in >= 0 and t_in < T:
            # Load RMS as half and convert to float32
            rms_in = tl.load(
                rms_ptr + b * stride_rms_b + t_in * stride_rms_t + g * stride_rms_g
            ).to(tl.float32)
            inv_rms = 1.0 / rms_in
            factor = w_norm_vals * inv_rms
            # Load input value as half and convert
            val_in = tl.load(
                value_ptr
                + b * stride_vb
                + t_in * stride_vt
                + g * stride_vg
                + c_idx * stride_vc,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            norm_val = val_in * factor
            # Load convolution weight as half and convert
            ch = g * C + c_idx
            w_conv = tl.load(
                conv_weight_ptr + ch * stride_w_ch + k * stride_w_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += norm_val * w_conv

    # Activation (SiLU)
    if activation:
        acc = acc * tl.sigmoid(acc)

    # Load current value for residual connection
    val_cur = tl.load(
        value_ptr + b * stride_vb + t * stride_vt + g * stride_vg + c_idx * stride_vc,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = val_cur + acc
    tl.store(
        output_ptr + b * stride_ob + t * stride_ot + g * stride_og + c_idx * stride_oc,
        out,
        mask=mask,
    )


# ----------------------------------------------------------------------
#  Main Module
# ----------------------------------------------------------------------
class EngramTri(nn.Module):
    def __init__(
        self,
        layer_id: int,
        engram_hidden_size: int,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 3,
        norm_eps: float = 1e-6,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        torch.manual_seed(41)
        super().__init__()

        self.layer_id = layer_id
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=[
                x
                for y in self.hash_mapping.vocab_size_across_layers[self.layer_id]
                for x in y
            ],
            D=engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )

        self.value_proj = nn.Linear(engram_hidden_size, hidden_size, device=device)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, hidden_size, device=device)
                for _ in range(hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(hidden_size, device=device) for _ in range(hc_mult)]
        )
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(hidden_size, device=device) for _ in range(hc_mult)]
        )
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            device="cuda",
        )
        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=1e-6, device="cuda") for _ in range(hc_mult)]
        )
        self.act_fn = nn.SiLU()

        # Precompute half‑precision buffers for efficiency
        self._precompute_buffers()

    def _precompute_buffers(self):
        # Key projection weights and biases (concatenated)
        key_weights = torch.cat([proj.weight for proj in self.key_projs], dim=0)
        key_biases = torch.cat([proj.bias for proj in self.key_projs], dim=0)
        self.register_buffer("key_proj_weight", key_weights)
        self.register_buffer("key_proj_bias", key_biases)

        # Value projection weights and biases (half)
        self.register_buffer("value_proj_weight", self.value_proj.weight)
        if self.value_proj.bias is not None:
            self.register_buffer("value_proj_bias", self.value_proj.bias)
        else:
            self.register_buffer("value_proj_bias", None)

        # RMSNorm weights
        w_key_norm = torch.stack(
            [self.norm1[i].weight for i in range(self.hc_mult)], dim=0
        )  # .half()
        w_query_norm = torch.stack(
            [self.norm2[i].weight for i in range(self.hc_mult)], dim=0
        )  # .half()
        w_norm = torch.stack(
            [self.norms[i].weight for i in range(self.hc_mult)], dim=0
        )  # .half()
        self.register_buffer("w_key_norm", w_key_norm)
        self.register_buffer("w_query_norm", w_query_norm)
        self.register_buffer("w_norm", w_norm)

        # Conv weight (squeezed to 2D)
        conv_weight = self.conv.weight.squeeze(1).contiguous()  # .half()
        self.register_buffer("conv_weight", conv_weight)

    def forward(self, input_ids, hidden_states):
        B, T, G, C = hidden_states.shape
        G = backbone_config.hc_mult
        C = backbone_config.hidden_size

        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids)[self.layer_id]
        ).cuda()
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        E = embeddings.shape[-1]

        # Key projection (fused)
        key_all = F.linear(
            embeddings,
            self.key_proj_weight,
            self.key_proj_bias,
        )  # (B, T, G*C)
        key_all = key_all.view(B, T, G, C).contiguous()

        # Query is hidden_states (already half)
        query_all = hidden_states  # (B, T, G, C)

        # Value projection
        value_proj_out = F.linear(
            embeddings, self.value_proj_weight, self.value_proj_bias
        )  # (B, T, C)
        value_proj_out = value_proj_out.contiguous()

        # Allocate intermediate tensors (half)
        value = torch.empty((B, T, G, C), device=embeddings.device, dtype=torch.float32)
        rms = torch.empty((B, T, G), device=embeddings.device, dtype=torch.float32)

        # Launch gate kernel
        sqrt_C = math.sqrt(C)
        eps = 1e-6  # RMSNorm default eps
        norm_eps = self.norms[0].eps

        grid_gate = (B * T * G,)
        engram_gate_value_rms_kernel[grid_gate](
            key_all,
            query_all,
            value_proj_out,
            self.w_key_norm,
            self.w_query_norm,
            value,
            rms,
            B,
            T,
            G,
            C,
            key_all.stride(0),
            key_all.stride(1),
            key_all.stride(2),
            key_all.stride(3),
            query_all.stride(0),
            query_all.stride(1),
            query_all.stride(2),
            query_all.stride(3),
            value_proj_out.stride(0),
            value_proj_out.stride(1),
            value_proj_out.stride(2),
            self.w_key_norm.stride(0),
            self.w_key_norm.stride(1),
            self.w_query_norm.stride(0),
            self.w_query_norm.stride(1),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            rms.stride(0),
            rms.stride(1),
            rms.stride(2),
            eps,
            sqrt_C,
            norm_eps,
        )

        # Fused conv + activation + residual
        total_channels = G * C
        K = self.conv.kernel_size[0]
        dilation = self.conv.dilation[0]
        conv_weight = self.conv_weight  # (total_channels, K)

        output = torch.empty(
            (B, T, G, C), device=embeddings.device, dtype=torch.float32
        )

        def grid_conv(args):
            return (
                args["B"]
                * args["T"]
                * args["G"]
                * ((args["C"] + args["BLOCK_SIZE_C"] - 1) // args["BLOCK_SIZE_C"]),
            )

        engram_conv_residual_kernel[grid_conv](
            value,
            rms,
            self.w_norm,
            conv_weight,
            output,
            B,
            T,
            G,
            C,
            K,
            dilation,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            rms.stride(0),
            rms.stride(1),
            rms.stride(2),
            self.w_norm.stride(0),
            self.w_norm.stride(1),
            conv_weight.stride(0),
            conv_weight.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            self.activation,
        )

        return output


class TransformerBlockTri(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.attn = lambda x: x
        self.moe = lambda x: x
        self.engram_tri = None
        if layer_id in engram_cfg.layer_ids:
            self.engram_tri = EngramTri(
                layer_id, engram_cfg.hidden_size, backbone_config.hidden_size
            )

    def forward(self, input_ids, hidden_states):
        if self.engram_tri is not None:
            hidden_states = (
                self.engram_tri(hidden_states=hidden_states, input_ids=input_ids)
                + hidden_states
            )
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states
