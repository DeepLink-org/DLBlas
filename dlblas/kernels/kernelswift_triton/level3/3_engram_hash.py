import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def engram_hash_kernel(
    ngram_token_ids_ptr,  # int32 [T, N]
    multipliers_ptr,      # int64 [L, N]
    vocab_sizes_ptr,      # int32 [L, N-1, E_PER]
    offsets_ptr,          # int32 [L, E_TOT]
    out_ptr,              # int32 [L, T, E_TOT]
    T: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    E_PER: tl.constexpr,
    E_TOT: tl.constexpr,
    stride_t0, stride_t1,
    stride_m0, stride_m1,
    stride_vs0, stride_vs1, stride_vs2,
    stride_off0, stride_off1,
    stride_out0, stride_out1, stride_out2,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    pid_l = tl.program_id(2)
    l = pid_l

    # Token tile indices
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # Column tile base
    e_base = pid_e * BLOCK_E

    # Base pointers for this layer
    out_layer_base = out_ptr + l * stride_out0
    mult_layer_base = multipliers_ptr + l * stride_m0
    vs_layer_base = vocab_sizes_ptr + l * stride_vs0
    off_layer_base = offsets_ptr + l * stride_off0

    # Initialize hash with gram 0
    t0_ptrs = ngram_token_ids_ptr + t_offsets * stride_t0 + 0 * stride_t1
    tok0 = tl.load(t0_ptrs, mask=mask_t, other=0).to(tl.int64)
    m0 = tl.load(mult_layer_base + 0 * stride_m1)  # int64 scalar
    h = tok0 * m0  # [BLOCK_T] int64

    # Iterate over subsequent grams; unrolled for performance
    for i in tl.static_range(1, N):
        ti_ptrs = ngram_token_ids_ptr + t_offsets * stride_t0 + i * stride_t1
        toki = tl.load(ti_ptrs, mask=mask_t, other=0).to(tl.int64)
        mi = tl.load(mult_layer_base + i * stride_m1)  # int64 scalar
        h = h ^ (toki * mi)

        # For current gram i, only one column per embed j exists in this tile at most.
        # We compute and store those columns directly to avoid computing modulo for all E lanes.
        for j in tl.static_range(0, E_PER):
            col_global = (i - 1) * E_PER + j  # in [0, E_TOT)
            col_local = col_global - e_base
            present = (col_local >= 0) & (col_local < BLOCK_E)
            if present:
                # Load divisor and offset scalars
                vs_i32 = tl.load(vs_layer_base + (i - 1) * stride_vs1 + j * stride_vs2)
                vs_i64 = vs_i32.to(tl.int64)
                off_i32 = tl.load(off_layer_base + col_global * stride_off1)

                # Python-style modulo with positive divisor
                rem = h % vs_i64
                rem = tl.where(rem < 0, rem + vs_i64, rem)

                vals = rem.to(tl.int32) + off_i32  # broadcast add
                # Store this column [BLOCK_T] into [L, T, E_TOT] at column = col_global
                out_col_ptrs = out_layer_base + t_offsets * stride_out1 + col_global * stride_out2
                tl.store(out_col_ptrs, vals, mask=mask_t)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        ngram_token_ids: torch.Tensor,
        multipliers: torch.Tensor,
        vocab_sizes: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triton-accelerated implementation matching the reference PyTorch semantics.
        Fallbacks to PyTorch when Triton/GPU is unavailable or when output is empty.
        """
        # Shapes
        num_tokens, max_ngram_size = ngram_token_ids.shape
        num_ngram_layers = multipliers.shape[0]
        assert multipliers.shape[1] == max_ngram_size
        assert vocab_sizes.shape[0] == num_ngram_layers
        assert vocab_sizes.shape[1] == max_ngram_size - 1
        num_embed_table_per_ngram = vocab_sizes.shape[2]
        E_TOT = (max_ngram_size - 1) * num_embed_table_per_ngram

        # If there is nothing to compute on the last dimension, short-circuit using reference behavior
        if E_TOT == 0:
            # Build empty output and add offsets (broadcast) to match reference
            out = torch.empty((num_ngram_layers, num_tokens, 0), dtype=torch.int32, device=ngram_token_ids.device)
            return out + offsets.unsqueeze(1)

        # Conditions for using Triton
        use_triton = (
            ngram_token_ids.is_cuda
            and multipliers.is_cuda
            and vocab_sizes.is_cuda
            and offsets.is_cuda
            and (ngram_token_ids.device == multipliers.device == vocab_sizes.device == offsets.device)
        )

        if not use_triton:
            # Reference PyTorch path
            prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)
            ans = [[] for _ in range(num_ngram_layers)]
            hashes = prod[:, :, 0].clone()
            for i in range(1, max_ngram_size):
                hashes.bitwise_xor_(prod[:, :, i])
                for layer_idx in range(num_ngram_layers):
                    ans[layer_idx].append(
                        (hashes[layer_idx].unsqueeze(-1) % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)).to(torch.int32)
                    )
            for layer_idx in range(num_ngram_layers):
                ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)
            output = torch.stack(ans, dim=0)
            return output + offsets.unsqueeze(1)

        # Triton-accelerated path
        L = num_ngram_layers
        T = num_tokens
        N = max_ngram_size
        E_PER = num_embed_table_per_ngram

        # Allocate output
        out = torch.empty((L, T, E_TOT), dtype=torch.int32, device=ngram_token_ids.device)

        # Extract strides (in elements)
        stride_t0, stride_t1 = ngram_token_ids.stride()
        stride_m0, stride_m1 = multipliers.stride()
        stride_vs0, stride_vs1, stride_vs2 = vocab_sizes.stride()
        stride_off0, stride_off1 = offsets.stride()
        stride_out0, stride_out1, stride_out2 = out.stride()

        # Grid and launch parameters
        BLOCK_T = 128
        BLOCK_E = 64
        grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(E_TOT, BLOCK_E), L)

        engram_hash_kernel[grid](
            ngram_token_ids,
            multipliers,
            vocab_sizes,
            offsets,
            out,
            T, N, L, E_PER, E_TOT,
            stride_t0, stride_t1,
            stride_m0, stride_m1,
            stride_vs0, stride_vs1, stride_vs2,
            stride_off0, stride_off1,
            stride_out0, stride_out1, stride_out2,
            BLOCK_T=BLOCK_T,
            BLOCK_E=BLOCK_E,
            num_warps=4,
            num_stages=2,
        )
        return out


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """Compute exclusive prefix-sum offsets from vocab_sizes.
    Args:
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
    Returns:
        Offsets of shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device), flat[:-1].cumsum(0, dtype=torch.int32)])
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def generate_test_data(params):
    num_tokens = params['num_tokens']
    max_ngram_size = params['ngram']
    num_ngram_layers = params['layers']
    num_embed_table_per_ngram = params['tables']
    ngram_token_ids = torch.randint(0, 100000, (num_tokens, max_ngram_size), dtype=torch.int32)
    multipliers = torch.randint(0, 100000, (num_ngram_layers, max_ngram_size), dtype=torch.int64)
    vocab_sizes = torch.randint(100000, 1000000, (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), dtype=torch.int32)
    offsets = make_offsets(vocab_sizes)
    return (ngram_token_ids, multipliers, vocab_sizes, offsets)


def test_engram_hash():
    return Model(*get_init_inputs()).forward(*get_inputs())


def get_inputs():
    params = {'num_tokens': 4096}
    num_tokens = params['num_tokens']
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8
    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(
        {'num_tokens': num_tokens, 'ngram': max_ngram_size, 'layers': num_ngram_layers, 'tables': num_embed_table_per_ngram})
    return [ngram_token_ids, multipliers, vocab_sizes, offsets]


def get_init_inputs():
    return []