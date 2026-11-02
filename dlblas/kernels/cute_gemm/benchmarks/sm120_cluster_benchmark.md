# SM120 Cluster Support Benchmark Results

**GPU**: NVIDIA GeForce RTX 5090, SM 12.0, 32 GB GDDR7, 170 SMs, 99 KB SMEM/block

## Summary

SM120 now supports cluster sizes up to 8 for reduction kernels (previously disabled).
For fp32 on SM120, tighter clustering thresholds are used to fit within the 99 KB SMEM limit.

## Benchmark Results (M=4096)

All bandwidth numbers in GB/s. Measured with `triton.testing.do_bench`.

I/O accounting (db = dtype bytes):
- **rmsnorm fwd**: read x + read w + write out = `2*M*N*db + N*4`
- **rmsnorm fwd+res**: read x + read res + write out + write res_out = `4*M*N*db + N*4`
- **rmsnorm bwd**: read x + read dy + read w + write dx = `3*M*N*db + N*4`
- **softmax fwd**: read x + write out = `2*M*N*db`
- **softmax bwd**: read dy + read y + write dx = `3*M*N*db`
- **CE fwd**: read x + read target + write loss = `M*N*db + M*12`
- **CE bwd**: read x + read target + read dloss + read lse + write dx = `2*M*N*db + M*16`

### BFloat16

| N | rmsnorm fwd | rmsnorm fwd+res | rmsnorm bwd | softmax fwd | softmax bwd | CE fwd | CE bwd |
|---|---|---|---|---|---|---|---|
| 4096 | 1477 | 1480 | 1292 | 1552 | 1473 | 1289 | 1485 |
| 8192 | 1531 | 1512 | 1441 | 1527 | 1542 | 1402 | 1482 |
| 16384 | 1539 | 1480 | 1474 | 1504 | 1555 | 1453 | 1495 |
| 32768 | 1520 | 1521 | 1370 | 1523 | 1568 | 1517 | 1466 |
| 65536 | 1521 | 1524 | 1387 | 1528 | 1572 | 1579 | 1487 |
| 131072 | 1524 | 1528 | SMEM | 1531 | 1577 | 1617 | 1497 |

### Float32

| N | rmsnorm fwd | rmsnorm fwd+res | rmsnorm bwd | softmax fwd | softmax bwd | CE fwd | CE bwd |
|---|---|---|---|---|---|---|---|
| 4096 | 1488 | 1500 | 1324 | 1481 | 1526 | 1373 | 1471 |
| 8192 | 1522 | 1521 | 1436 | 1500 | 1554 | 1463 | 1501 |
| 16384 | 1521 | 1524 | 1396 | 1513 | 1567 | 1517 | 1517 |
| 32768 | 1523 | 1523 | 1468 | 1523 | 1572 | 1575 | 1515 |
| 65536 | 1525 | 1526 | SMEM | 1525 | 1575 | 1610 | 1504 |
| 131072 | 1530 | SMEM | SMEM | 1530 | SMEM | 1642 | 1512 |

SMEM = exceeds 99 KB shared memory even at cluster_n=8.

## SM120 fp32 Cluster Thresholds

To fit within 99 KB SMEM, fp32 on SM12x uses tighter clustering thresholds:

| Kernel | Default fp32 thresholds | SM12x fp32 thresholds | Reason |
|---|---|---|---|
| CE / Softmax fwd | (32K,1) (64K,2) (128K,4) (256K,8) | (16K,1) (32K,2) (64K,4) (128K,8) | 1 SMEM tensor |
| Softmax bwd | (16K,1) (32K,2) (64K,4) (128K,8) | (8K,1) (16K,2) (32K,4) (64K,8) | 2 SMEM tensors |
| RMSNorm fwd | (32K,1) (64K,2) (128K,4) (256K,8) | (8K,1) (16K,2) (32K,4) (64K,8) | conservative for residual |
| RMSNorm bwd | (8K,1) (16K,2) (32K,4) (64K,8) | (1K,1) (8K,2) (16K,4) (32K,8) | 2 tensors x 2 stages |

## Max Active Clusters

| cluster_size | max_active_clusters |
|---|---|
| 1 | 170 |
| 2 | 85 |
| 4 | 41 |
| 8 | 19 |

## SMEM Limits on SM120 (99 KB, max cluster_n=8)

fp32 max N per kernel (with adjusted thresholds):

| Kernel | fp32 max N | bf16 max N |
|---|---|---|
| CE fwd (1 tensor) | 131072 | all |
| Softmax fwd (1 tensor) | 131072 | all |
| Softmax bwd (2 tensors) | 65536 | 131072 |
| RMSNorm fwd (1 tensor) | 131072 | all |
| RMSNorm fwd+res (2 tensors) | 65536 | 131072 |
| RMSNorm bwd (2 tensors x2 stages) | 32768 | 65536 |
