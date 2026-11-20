#!/usr/bin/env python3
"""Generate OneGenomeRice TP=4 scoring-shape operator cases.

EvalScope's 2K/4K/8K/16K random prompts become 4095/8191/16383/32767
OneGenomeRice tokens because the generated bases are separated by spaces.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLASH_DIR = ROOT / "test/zoo/teco/flash_attention/test_case"
RESHAPE_DIR = ROOT / "test/zoo/teco/reshape_and_cache/test_case"

Q_HEADS = 4
KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 32

SCORING_LENGTHS = {
    "t1": (4095, 4096),
    "t2": (8191, 8192),
    "t3": (16383, 16384),
    "t4": (32767, 32768),
}


def tensor(name, dims, dtype, seed, lower, upper):
    shape = "\n".join(f"    dims: {d}" for d in dims)
    return f"""input {{
  id: "{name}"
  shape {{
{shape}
  }}
  layout: LAYOUT_ARRAY
  dtype: {dtype}
  ttype: TENSOR
  random_data {{
    seed: {seed}
    lower_bound: {lower}
    upper_bound: {upper}
    distribution: UNIFORM
  }}
}}
"""


def seq_tensor(name, value):
    return f"""input {{
  id: "{name}"
  shape {{ dims: 1 }}
  layout: LAYOUT_ARRAY
  dtype: DTYPE_INT32
  ttype: TENSOR
  prev_value {{ value_i: {value} }}
}}
"""


def flash_case(q_len, kv_len):
    blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    return (
        'op_name: "flash_attention"\n'
        + tensor("blockTable", [1, blocks], "DTYPE_INT32", 100, 0, blocks)
        + tensor("qData", [q_len, Q_HEADS, HEAD_SIZE], "DTYPE_HALF", 1, -1.0, 1.0)
        + tensor(
            "kCache",
            [blocks, KV_HEADS, BLOCK_SIZE, HEAD_SIZE],
            "DTYPE_HALF",
            2,
            -1.0,
            1.0,
        )
        + tensor(
            "vCache",
            [blocks, KV_HEADS, BLOCK_SIZE, HEAD_SIZE],
            "DTYPE_HALF",
            3,
            -1.0,
            1.0,
        )
        + seq_tensor("q_seq_lens", q_len)
        + seq_tensor("kv_seq_lens", kv_len)
        + f"""output {{
  id: "oData"
  shape {{
    dims: {q_len}
    dims: {Q_HEADS}
    dims: {HEAD_SIZE}
  }}
  layout: LAYOUT_ARRAY
  dtype: DTYPE_HALF
  ttype: TENSOR
  reused: false
}}
tecokernel_param {{
  flash_attention_param {{
    max_seqlen_q: {q_len}
    max_seqlen_k: {kv_len}
  }}
}}
test_param {{
  error_func: DIFF1
  error_threshold: 2e-2
}}
"""
    )


def reshape_tensor(name, dims, dtype, seed, reused=False):
    kind = "output" if reused else "input"
    shape = "\n".join(f"    dims: {d}" for d in dims)
    reused_line = "  reused: true\n" if reused else ""
    return f"""{kind} {{
  id: "{name}"
  shape {{
{shape}
  }}
{reused_line}  layout: LAYOUT_NCHW
  dtype: {dtype}
  ttype: TENSOR
  random_data {{
    seed: {seed}
    lower_bound: -1.0
    upper_bound: 1.0
    distribution: UNIFORM
  }}
}}
"""


def reshape_case(num_tokens, num_blocks):
    return (
        'op_name: "reshape_and_cache"\n'
        + reshape_tensor("key", [num_tokens, KV_HEADS, HEAD_SIZE], "DTYPE_HALF", 42)
        + reshape_tensor("value", [num_tokens, KV_HEADS, HEAD_SIZE], "DTYPE_HALF", 43)
        + tensor("slot_mapping", [num_tokens], "DTYPE_INT64", 44, 0, num_blocks * BLOCK_SIZE)
        + reshape_tensor(
            "key_cache",
            [num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_SIZE],
            "DTYPE_HALF",
            45,
            reused=True,
        )
        + reshape_tensor(
            "value_cache",
            [num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_SIZE],
            "DTYPE_HALF",
            46,
            reused=True,
        )
        + "tecokernel_param { reshape_and_cache_param {} }\n"
    )


def main():
    FLASH_DIR.mkdir(parents=True, exist_ok=True)
    RESHAPE_DIR.mkdir(parents=True, exist_ok=True)

    for name, (prompt_len, decode_len) in SCORING_LENGTHS.items():
        (FLASH_DIR / f"scoring_decode_{name}_q1_k{decode_len}.prototxt").write_text(
            flash_case(1, decode_len)
        )
        (FLASH_DIR / f"scoring_prefill_{name}_q2048_k{prompt_len}.prototxt").write_text(
            flash_case(2048, prompt_len)
        )
        (FLASH_DIR / f"scoring_tail_{name}_q2047_k{prompt_len}.prototxt").write_text(
            flash_case(2047, prompt_len)
        )

    # Exercise non-divisible split counts in the 16-way FlashDecoding path.
    # These cover a partial 17th block, 33 full blocks, and a 1025-block
    # case above the compact decode launch limit.
    decode_boundaries = {
        "b17_tail": 513,
        "b33": 1056,
        "b1025": 32800,
    }
    for name, decode_len in decode_boundaries.items():
        (FLASH_DIR / f"scoring_decode_{name}_q1_k{decode_len}.prototxt").write_text(
            flash_case(1, decode_len)
        )

    # 6400 blocks covers the largest prompt cache used by T4.
    (RESHAPE_DIR / "scoring_decode_tokens1_cache6400.prototxt").write_text(
        reshape_case(1, 6400)
    )
    (RESHAPE_DIR / "scoring_prefill_tokens2048_cache6400.prototxt").write_text(
        reshape_case(2048, 6400)
    )


if __name__ == "__main__":
    main()
