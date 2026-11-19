#!/usr/bin/env python3
import json, math, os, sys
import torch
import torch_sdaa  # noqa
import tecoops

sys.path.insert(0, ".")
from python_api_test.test_flash_attention import (
    _make_random_fa_inputs, cdiv, flash_attn_varlen_func_ref,
    sdaa_kv_to_ref_layout,
)

os.environ["TECOOPS_SPLIT_KV"] = "1"
torch.manual_seed(20260808)
results = []
for seq_len in (1024, 1025, 4095, 8191, 16383, 32767):
    q, k, v, block_table = _make_random_fa_inputs(
        [1], [seq_len], 4, 2, 64, 32, cdiv(seq_len, 32) + 8
    )
    ref = flash_attn_varlen_func_ref(
        q, sdaa_kv_to_ref_layout(k), sdaa_kv_to_ref_layout(v),
        [1], [seq_len], block_table, True, 1.0 / math.sqrt(64.0)
    )
    qd, kd, vd, bd = [x.contiguous().to("sdaa") for x in (q, k, v, block_table)]
    out = torch.empty_like(qd)
    tecoops.flash_attn_varlen_func(
        qd, kd, vd, 1, torch.tensor([0, 1], dtype=torch.int32).to("sdaa"),
        seq_len, torch.Tensor(), torch.tensor([seq_len], dtype=torch.int32).to("sdaa"),
        1.0 / math.sqrt(64.0), True, torch.Tensor(), bd, False, out,
    )
    err = (out.cpu().float() - ref.float()).abs()
    item = {"seq_len": seq_len, "max_abs": err.max().item(),
            "mean_abs": err.mean().item(), "finite": bool(torch.isfinite(err).all())}
    item["passed"] = item["finite"] and item["max_abs"] <= 0.003 and item["mean_abs"] <= 0.0005
    results.append(item)
    torch.sdaa.empty_cache()
payload = {"shape": "batch1/q1/qh4/kvh2/d64/block32", "results": results,
           "passed": all(x["passed"] for x in results)}
print(json.dumps(payload, indent=2))
raise SystemExit(0 if payload["passed"] else 1)
