# ============================================================================
# MTPBlock 测试数据生成脚本 - K4 hc_post + K2 hc_pre
# ============================================================================

import numpy as np
import os
import sys

from golden import compute_golden_k1_embed_fuse, compute_golden_k2_hc_pre, compute_golden_k4_hc_post
from golden import compute_golden_k3_attn_block, compute_golden_k5_moe_block, compute_golden_k6_mtp_head

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Demo 参数
b, s, hc, d = 1, 8, 4, 512
mix_hc = (2 + hc) * hc  # 24
hcd = hc * d  # 2048

# ============================================================================
# K4 hc_post 数据生成
# ============================================================================

# x: [b, s, d] bf16
x_k4 = np.random.randn(b, s, d).astype(np.float32) * 0.5
x_k4_bf16 = x_k4.astype(np.float16)

# residual: [b, s, hc, d] bf16
residual = np.random.randn(b, s, hc, d).astype(np.float32) * 0.5
residual_bf16 = residual.astype(np.float16)

# post: [b, s, hc] fp32
post = np.random.randn(b, s, hc).astype(np.float32) * 0.5

# comb: [b, s, hc, hc] fp32 (doubly stochastic)
comb = np.random.randn(b, s, hc, hc).astype(np.float32) * 0.1
comb = np.exp(comb) / np.exp(comb).sum(axis=-1, keepdims=True)
comb = comb / comb.sum(axis=-2, keepdims=True)

x_k4_bf16.tofile("input/k4_x.bin")
residual_bf16.tofile("input/k4_residual.bin")
post.tofile("input/k4_post.bin")
comb.tofile("input/k4_comb.bin")

golden_k4 = compute_golden_k4_hc_post(x_k4, residual, post, comb)
golden_k4_bf16 = golden_k4.astype(np.float16)
golden_k4_bf16.tofile("output/golden_k4.bin")

print(f"K4 hc_post test data generated:")
print(f"  x:         {x_k4.shape}        (bf16)")
print(f"  residual:  {residual.shape}  (bf16)")
print(f"  post:      {post.shape}      (fp32)")
print(f"  comb:      {comb.shape}      (fp32)")
print(f"  golden:    {golden_k4.shape}    (bf16)")

# ============================================================================
# K2 hc_pre 数据生成
# ============================================================================

# x: [b, s, hc, d] bf16 (HC expanded)
x_k2 = np.random.randn(b, s, hc, d).astype(np.float32) * 0.5
x_k2_bf16 = x_k2.astype(np.float16)

# hc_fn: [mix_hc, hc*d] fp32
hc_fn = np.random.randn(mix_hc, hcd).astype(np.float32) * 0.01

# hc_scale: [3] fp32
hc_scale = np.ones(3, dtype=np.float32)

# hc_base: [mix_hc] fp32
hc_base = np.zeros(mix_hc, dtype=np.float32)

x_k2_bf16.tofile("input/k2_x.bin")
hc_fn.tofile("input/k2_hc_fn.bin")
hc_scale.tofile("input/k2_hc_scale.bin")
hc_base.tofile("input/k2_hc_base.bin")

golden_y, golden_pre, golden_post, golden_comb = compute_golden_k2_hc_pre(
    x_k2, hc_fn, hc_scale, hc_base)

golden_y.astype(np.float16).tofile("output/golden_k2_y.bin")
golden_pre.astype(np.float32).tofile("output/golden_k2_pre.bin")
golden_post.astype(np.float32).tofile("output/golden_k2_post.bin")
golden_comb.astype(np.float32).tofile("output/golden_k2_comb.bin")

print(f"\nK2 hc_pre test data generated:")
print(f"  x:          {x_k2.shape}         (bf16)")
print(f"  hc_fn:      {hc_fn.shape}       (fp32)")
print(f"  hc_scale:   {hc_scale.shape}          (fp32)")
print(f"  hc_base:    {hc_base.shape}         (fp32)")
print(f"  y golden:   {golden_y.shape}          (bf16)")
print(f"  pre golden: {golden_pre.shape}       (fp32)")
print(f"  post golden:{golden_post.shape}       (fp32)")
print(f"  comb golden:{golden_comb.shape}     (fp32)")

# ============================================================================
# K1 embed_fuse 数据生成
# ============================================================================

test_vocab = 1000  # small vocab for testing

x_k1 = np.random.randn(b, s, hc, d).astype(np.float32) * 0.5
x_k1_bf16 = x_k1.astype(np.float16)

input_ids = np.random.randint(0, test_vocab, (b, s)).astype(np.int64)

embed_w = np.random.randn(test_vocab, d).astype(np.float32) * 0.1
embed_w_bf16 = embed_w.astype(np.float16)

enorm_w = np.ones(d, dtype=np.float32)
e_proj_w = np.random.randn(d, d).astype(np.float32) * 0.02
e_proj_w_bf16 = e_proj_w.astype(np.float16)
h_proj_w = np.random.randn(d, d).astype(np.float32) * 0.02
h_proj_w_bf16 = h_proj_w.astype(np.float16)
hnorm_w = np.ones(d, dtype=np.float32)

x_k1_bf16.tofile("input/k1_x.bin")
input_ids.tofile("input/k1_ids.bin")
embed_w_bf16.tofile("input/k1_embed_w.bin")
enorm_w.tofile("input/k1_enorm_w.bin")
e_proj_w_bf16.tofile("input/k1_eproj_w.bin")
h_proj_w_bf16.tofile("input/k1_hproj_w.bin")
hnorm_w.tofile("input/k1_hnorm_w.bin")

golden_k1 = compute_golden_k1_embed_fuse(
    x_k1, input_ids, embed_w, enorm_w, e_proj_w, h_proj_w, hnorm_w)
golden_k1_bf16 = golden_k1.astype(np.float16)
golden_k1_bf16.tofile("output/golden_k1.bin")

print(f"\nK1 embed_fuse test data generated:")
print(f"  x:          {x_k1.shape}         (bf16)")
print(f"  ids:        {input_ids.shape}          (int64)")
print(f"  embed_w:    [{test_vocab}, {d}]      (bf16)")
print(f"  golden:     {golden_k1.shape}     (bf16)")

# ============================================================================
# K3 attn_block 数据生成
# ============================================================================

n_heads = 8
head_dim = 64
q_lora = 256
o_lora = 128
n_groups = 2
nhd = n_heads * head_dim
hpg = nhd // n_groups  # heads per group = 256
softmax_scale = 1.0 / np.sqrt(head_dim)  # 0.125

# x: [s, d] bf16
x_k3 = np.random.randn(s, d).astype(np.float32) * 0.5
x_k3_bf16 = x_k3.astype(np.float16)

# wq_a: [q_lora, d] bf16
wq_a = np.random.randn(q_lora, d).astype(np.float32) * 0.02
wq_a_bf16 = wq_a.astype(np.float16)

# q_norm: [q_lora] fp32
q_norm = np.ones(q_lora, dtype=np.float32)

# wq_b: [n_heads*head_dim, q_lora] bf16
wq_b = np.random.randn(nhd, q_lora).astype(np.float32) * 0.02
wq_b_bf16 = wq_b.astype(np.float16)

# wkv: [head_dim, d] bf16
wkv = np.random.randn(head_dim, d).astype(np.float32) * 0.02
wkv_bf16 = wkv.astype(np.float16)

# kv_norm: [head_dim] fp32
kv_norm = np.ones(head_dim, dtype=np.float32)

# wo_a: [n_groups, o_lora, n_heads*head_dim/n_groups] bf16
wo_a = np.random.randn(n_groups, o_lora, hpg).astype(np.float32) * 0.02
wo_a_bf16 = wo_a.astype(np.float16)

# wo_b: [d, n_groups*o_lora] bf16
wo_b = np.random.randn(d, n_groups * o_lora).astype(np.float32) * 0.02
wo_b_bf16 = wo_b.astype(np.float16)

# attn_sink: [n_heads] fp32
attn_sink = np.zeros(n_heads, dtype=np.float32)

x_k3_bf16.tofile("input/k3_x.bin")
wq_a_bf16.tofile("input/k3_wq_a.bin")
q_norm.tofile("input/k3_q_norm.bin")
wq_b_bf16.tofile("input/k3_wq_b.bin")
wkv_bf16.tofile("input/k3_wkv.bin")
kv_norm.tofile("input/k3_kv_norm.bin")
wo_a_bf16.tofile("input/k3_wo_a.bin")
wo_b_bf16.tofile("input/k3_wo_b.bin")
attn_sink.tofile("input/k3_attn_sink.bin")

golden_k3 = compute_golden_k3_attn_block(
    x_k3, wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink,
    softmax_scale)
golden_k3_bf16 = golden_k3.astype(np.float16)
golden_k3_bf16.tofile("output/golden_k3.bin")

print(f"\nK3 attn_block test data generated:")
print(f"  x:            {x_k3.shape}          (bf16)")
print(f"  wq_a:         {wq_a.shape}         (bf16)")
print(f"  q_norm:       {q_norm.shape}           (fp32)")
print(f"  wq_b:         {wq_b.shape}        (bf16)")
print(f"  wkv:          {wkv.shape}          (bf16)")
print(f"  kv_norm:      {kv_norm.shape}          (fp32)")
print(f"  wo_a:         {wo_a.shape}     (bf16)")
print(f"  wo_b:         {wo_b.shape}        (bf16)")
print(f"  attn_sink:    {attn_sink.shape}           (fp32)")
print(f"  golden:       {golden_k3.shape}          (bf16)")

# ============================================================================
# K5 moe_block 数据生成
# ============================================================================

n_experts = 8
topk = 2
inter_dim = 512
bs = b * s  # 8

# x: [b*s, d] bf16
x_k5 = np.random.randn(bs, d).astype(np.float32) * 0.5
x_k5_bf16 = x_k5.astype(np.float16)

# gate_weight: [n_experts, d] bf16 (unused in demo)
gate_w = np.random.randn(n_experts, d).astype(np.float32) * 0.01
gate_w_bf16 = gate_w.astype(np.float16)

# gate_bias: [n_experts] fp32
gate_b = np.zeros(n_experts, dtype=np.float32)

# shared expert w1: [inter_dim, d] bf16
sw1 = np.random.randn(inter_dim, d).astype(np.float32) * 0.02 / np.sqrt(d)
sw1_bf16 = sw1.astype(np.float16)

# shared expert w2: [d, inter_dim] bf16
sw2 = np.random.randn(d, inter_dim).astype(np.float32) * 0.02 / np.sqrt(inter_dim)
sw2_bf16 = sw2.astype(np.float16)

# shared expert w3: [inter_dim, d] bf16
sw3 = np.random.randn(inter_dim, d).astype(np.float32) * 0.02 / np.sqrt(d)
sw3_bf16 = sw3.astype(np.float16)

x_k5_bf16.tofile("input/k5_x.bin")
gate_w_bf16.tofile("input/k5_gate_w.bin")
gate_b.tofile("input/k5_gate_b.bin")
sw1_bf16.tofile("input/k5_sw1.bin")
sw2_bf16.tofile("input/k5_sw2.bin")
sw3_bf16.tofile("input/k5_sw3.bin")

golden_k5 = compute_golden_k5_moe_block(
    x_k5, gate_w, gate_b, sw1, sw2, sw3)
golden_k5_bf16 = golden_k5.astype(np.float16)
golden_k5_bf16.tofile("output/golden_k5.bin")

print(f"\nK5 moe_block test data generated:")
print(f"  x:            {x_k5.shape}          (bf16)")
print(f"  gate_w:       {gate_w.shape}         (bf16)")
print(f"  gate_b:       {gate_b.shape}           (fp32)")
print(f"  sw1:          {sw1.shape}        (bf16)")
print(f"  sw2:          {sw2.shape}        (bf16)")
print(f"  sw3:          {sw3.shape}        (bf16)")
print(f"  golden:       {golden_k5.shape}          (bf16)")

# ============================================================================
# K6 mtp_head 数据生成
# ============================================================================

test_vocab_k6 = 1000  # small vocab for testing

# x: [b, s, hc, d] bf16
x_k6 = np.random.randn(b, s, hc, d).astype(np.float32) * 0.5
x_k6_bf16 = x_k6.astype(np.float16)

# hc_head_fn: [hc, hc*d] fp32
hc_head_fn_k6 = np.random.randn(hc, hcd).astype(np.float32) * 0.01

# hc_head_scale: [1] fp32
hc_head_scale_k6 = np.ones(1, dtype=np.float32)

# hc_head_base: [hc] fp32
hc_head_base_k6 = np.zeros(hc, dtype=np.float32)

# norm_weight: [d] fp32
norm_w_k6 = np.ones(d, dtype=np.float32)

# head_weight: [vocab, d] fp32
head_w_k6 = np.random.randn(test_vocab_k6, d).astype(np.float32) * 0.01

x_k6_bf16.tofile("input/k6_x.bin")
hc_head_fn_k6.tofile("input/k6_hc_fn.bin")
hc_head_scale_k6.tofile("input/k6_hc_scale.bin")
hc_head_base_k6.tofile("input/k6_hc_base.bin")
norm_w_k6.tofile("input/k6_norm_w.bin")
head_w_k6.tofile("input/k6_head_w.bin")

golden_k6 = compute_golden_k6_mtp_head(
    x_k6, hc_head_fn_k6, hc_head_scale_k6, hc_head_base_k6,
    norm_w_k6, head_w_k6)
golden_k6.tofile("output/golden_k6.bin")  # fp32 logits

print(f"\nK6 mtp_head test data generated:")
print(f"  x:            {x_k6.shape}         (bf16)")
print(f"  hc_fn:        {hc_head_fn_k6.shape}       (fp32)")
print(f"  hc_scale:     {hc_head_scale_k6.shape}           (fp32)")
print(f"  hc_base:      {hc_head_base_k6.shape}           (fp32)")
print(f"  norm_w:       {norm_w_k6.shape}           (fp32)")
print(f"  head_w:       {head_w_k6.shape}      (fp32)")
print(f"  golden:       {golden_k6.shape}       (fp32)")
