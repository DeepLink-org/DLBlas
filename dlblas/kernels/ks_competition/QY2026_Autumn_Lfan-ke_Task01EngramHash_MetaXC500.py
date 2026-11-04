"""KernelSwift Task01 engram_hash for MetaX C500: one fused kernel, one thread per (layer, token) row."""

import os

import torch
from torch.utils.cpp_extension import load_inline

NUM_TOKENS = 4096
MAX_NGRAM_SIZE = 3
NUM_NGRAM_LAYERS = 2
NUM_TABLES = 8

DEFAULT_BLOCK = 256


def _cuda_source():
    return r"""
#include <torch/extension.h>
#include <cstdint>

__global__ void engram_hash_kernel(
    const int32_t* __restrict__ tok_ids,   // [T, G]
    const int64_t* __restrict__ mult,      // [L, G]
    const int32_t* __restrict__ vocab,     // [L, G-1, TB]
    const int32_t* __restrict__ offsets,   // [L, OC]
    int32_t* __restrict__ out,             // [L, T, OC]
    int T, int G, int TB, int rows) {
  const int OC = (G - 1) * TB;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    const int tok = row % T;
    const int l = row / T;
    const int32_t* trow = tok_ids + tok * G;
    const int64_t* mrow = mult + l * G;
    const int32_t* voc_l = vocab + l * (G - 1) * TB;
    const int32_t* off_l = offsets + l * OC;
    int32_t* orow = out + (long)row * OC;

    int64_t h = (int64_t)trow[0] * mrow[0];
    for (int s = 0; s < G - 1; ++s) {
      h ^= (int64_t)trow[s + 1] * mrow[s + 1];
      const int32_t* voc_s = voc_l + s * TB;
      const int32_t* off_s = off_l + s * TB;
      int32_t* out_s = orow + s * TB;
      for (int t = 0; t < TB; ++t) {
        out_s[t] = (int32_t)(h % (int64_t)voc_s[t]) + off_s[t];
      }
    }
  }
}

torch::Tensor engram_hash(torch::Tensor tok_ids_, torch::Tensor mult_,
                          torch::Tensor vocab_, torch::Tensor offsets_,
                          int64_t block) {
  // .contiguous() on every call costs about 6 us of dispatch.
  auto tok_ids = tok_ids_.is_contiguous() ? tok_ids_ : tok_ids_.contiguous();
  auto mult = mult_.is_contiguous() ? mult_ : mult_.contiguous();
  auto vocab = vocab_.is_contiguous() ? vocab_ : vocab_.contiguous();
  auto offsets = offsets_.is_contiguous() ? offsets_ : offsets_.contiguous();

  const int T = (int)tok_ids.size(0);
  const int G = (int)tok_ids.size(1);
  const int L = (int)mult.size(0);
  const int TB = (int)vocab.size(2);
  const int OC = (G - 1) * TB;
  const int rows = L * T;

  auto out = torch::empty({L, T, OC}, tok_ids.options().dtype(torch::kInt32));
  const int threads = (int)block;
  int want = (rows + threads - 1) / threads;
  const int blocks = want > 65535 ? 65535 : (want < 1 ? 1 : want);

  engram_hash_kernel<<<blocks, threads>>>(
      tok_ids.data_ptr<int32_t>(), mult.data_ptr<int64_t>(),
      vocab.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(), T, G, TB, rows);
  return out;
}
"""


def _mod(_cache=[]):
    """Lazy: auto_bench drops module-level statements."""
    if not _cache:
        _cache.append(
            load_inline(
                name="engram_hash_maca",
                cpp_sources=(
                    "torch::Tensor engram_hash(torch::Tensor tok_ids, torch::Tensor mult, "
                    "torch::Tensor vocab, torch::Tensor offsets, int64_t block);"
                ),
                cuda_sources=_cuda_source(),
                functions=["engram_hash"],
                # load_inline builds host code at -O0.
                extra_cflags=["-O3"],
                verbose=False,
            )
        )
    return _cache[0]


def _block_size():
    return int(os.environ.get("ENGRAM_HASH_BLOCK", DEFAULT_BLOCK))


class Model(torch.nn.Module):
    def forward(self, ngram_token_ids, multipliers, vocab_sizes, offsets):
        num_ngram_layers = multipliers.shape[0]
        max_ngram_size = multipliers.shape[1]
        prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)
        ans: list = [[] for _ in range(num_ngram_layers)]
        hashes = prod[:, :, 0].clone()
        for i in range(1, max_ngram_size):
            hashes.bitwise_xor_(prod[:, :, i])
            for layer_idx in range(num_ngram_layers):
                ans[layer_idx].append(
                    (
                        hashes[layer_idx].unsqueeze(-1)
                        % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)
                    ).to(torch.int32)
                )
        cols = [torch.cat(a, dim=-1) for a in ans]
        return torch.stack(cols, dim=0) + offsets.unsqueeze(1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._op = _mod().engram_hash
        self._block = _block_size()

    def forward(self, ngram_token_ids, multipliers, vocab_sizes, offsets):
        return self._op(ngram_token_ids, multipliers, vocab_sizes, offsets, self._block)


def _make_offsets(vocab_sizes):
    cols = []
    for layer in range(vocab_sizes.shape[0]):
        flat = vocab_sizes[layer].reshape(-1)
        zero = torch.zeros(1, dtype=torch.int32, device=flat.device)
        cols.append(torch.cat([zero, flat[:-1].cumsum(0, dtype=torch.int32)]))
    return torch.stack(cols, dim=0)


def get_init_inputs():
    return []


def get_inputs():
    device = "cuda"
    torch.manual_seed(233)
    ngram_token_ids = torch.randint(
        0, 100000, (NUM_TOKENS, MAX_NGRAM_SIZE), dtype=torch.int32, device=device
    )
    multipliers = torch.randint(
        0, 100000, (NUM_NGRAM_LAYERS, MAX_NGRAM_SIZE), dtype=torch.int64, device=device
    )
    vocab_sizes = torch.randint(
        100000,
        1000000,
        (NUM_NGRAM_LAYERS, MAX_NGRAM_SIZE - 1, NUM_TABLES),
        dtype=torch.int32,
        device=device,
    )
    return [ngram_token_ids, multipliers, vocab_sizes, _make_offsets(vocab_sizes)]
