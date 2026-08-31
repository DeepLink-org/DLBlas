"""Task01 engram_hash — 优化实现 (v1)：一个融合 MXMACA kernel 顶掉整段 torch 参考。

参考实现把 (L layer × G-1 ngram) 四次循环里的每一步都摊成独立的 torch 算子——int64 乘法、
bitwise_xor_、取模、to(int32)、逐层 cat、最后 stack 再加 offsets——十几次 kernel 启动加同样多
的中间张量往返显存，而真正的算术只有每个输出元素几次乘、异或、一次取模。计算量小到这个算子
完全是启动开销与访存 bound，所以优化的全部意义就是**把它压成一次 launch**。

kernel 布局：一个线程负责一整行 (layer, token)。行内的 OC = (G-1)*TB 个输出共用同一串前缀
异或，所以前缀在寄存器里只推进一次，随后把 OC 个结果连着写出——没有任何中间张量落显存。
grid-stride 让 blockDim 可调而不改语义。

★这个布局是量出来换的。先写的是「一线程一输出元素」：正确，但同一行的 16 个线程各自把前缀
异或重算一遍，int64 乘法冗余 16 倍，索引分解也重复 16 次。改成一线程一行后 0.0410 → 0.0377 ms
（裸调 8.00× → 8.71×）。同一台机器上还量了四件事，其中三件推翻了先验判断：

  - 把 int64 取模换成双精度倒数估商：**没有变快**（0.0400 vs 0.0393 ms）
  - 把取模整个去掉（诊断用，结果是错的）：只省 1.5 µs —— **取模根本不是瓶颈**
  - 改三维 grid 彻底消除索引除法：**更慢**（0.0410），launch 开销盖过省下的除法
  - 索引从 64 位改 32 位：有效（0.0393 → 0.0387），软件实现的 64 位整数除法确实贵

参考点：一个什么都不算的空 kernel 加一次同尺寸 `torch::empty`，这台机器上是 0.0242 ms。

★语义关键点：第 s 个输出切片用的是 prod[..., 0..s+1] 的**前缀**异或，不是全量异或。参考实现里
`hashes` 是循环外建、循环内原地 bitwise_xor_ 累积的，这一点极易看漏；ref_oracle.py 里留了一个
写成全量异或的变异守卫来钉住它。输出是 int32，harness 用 torch.equal 逐位比，没有容差。

★harness 约束：auto_bench.py 的 `_filter_module_ast` 只保留 import、class/def 和字面量赋值，
模块级的其它语句会被静默丢弃。所以扩展必须在函数里懒加载，不能写成模块级赋值——写成模块级
赋值时 forward 会在评测时报 `name '_mod' is not defined`。
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

NUM_TOKENS = 4096
MAX_NGRAM_SIZE = 3
NUM_NGRAM_LAYERS = 2
NUM_TABLES = 8

# 每个 block 的线程数。一线程一行的布局下 C500 上扫过 64/128/256/512，256 是甜点。
DEFAULT_BLOCK = 256


def _cuda_source():
    return r"""
#include <torch/extension.h>
#include <cstdint>

// 一线程一行 (layer, token)。行内的 OC = (G-1)*TB 个输出共用同一串前缀异或，
// 所以前缀只算一次，然后把 OC 个结果连着写出去。
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
      h ^= (int64_t)trow[s + 1] * mrow[s + 1];   // 前缀异或推进一项
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
  // 只在真的不连续时才拷贝：harness 传进来的本来就是连续张量，而无条件调 .contiguous()
  // 每次要多四趟 dispatch，实测吃掉约 6 µs（7.2x -> 8.4x）。
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
    """编译一次、缓存住的扩展。放在函数里是 harness 的硬要求（见模块头）。"""
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
        # 连续性检查在 C++ 侧做，见 engram_hash()：这里每多一次 python 侧 .contiguous()
        # 就多一趟 dispatch，而这个算子总共才 39 µs。
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
