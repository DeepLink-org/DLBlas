# Vendor-Opt Cross-Backend Performance Matrix

Generated: 2026-06-26T12:38:51Z | registry: /datapool/zmz/04kernelagent/waic/.humanize/vendor-runs/registry.jsonl | runs: 7

| operator | shape | dtype | baseline_us | metax-c500 |
|---|---|---|---|---|
| big_fuse | residual[1,512,4,1280]bf16_fn[24,5120] | bf16 | 531.0 | 373.0 (1.42x) |
| engram_gate_bwd | 14_4_128 | bf16 | 42.4 | 27.6 (1.53x) |
| engram_gate_w_reduce | 108_4_4096 | fp32_bf16 | 37.1 | 32.2 (1.15x) |
| engram_hash | num_tokens=4096,max_ngram_size=3,num_ngram_layers=2,num_embed_table_per_ngram=8 | int32 | 38.3 | 19.4 (1.98x) |
| hc_split_sinkhorn | B2_S8_HC4 | float32 | 144.1 | 38.4 (3.75x) |
| sinkhorn | 1_1024_4_4 | fp32 | 152.4 | 37.2 (4.09x) |
| sparse_attn | B=2,M=16,N=32,H=8,D=64,TopK=16 | bfloat16 | 65.5 | 45.8 (1.43x) |
