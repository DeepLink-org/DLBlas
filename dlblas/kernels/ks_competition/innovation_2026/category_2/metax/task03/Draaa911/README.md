# KernelSwift Task03 norm_fn

This implementation fuses the fixed-shape projection and RMS normalization in one MACA kernel. Each block computes three output channels, and a native CPython bridge reuses the output tensor while reading the current input pointers on every call.

## Environment

- CPython 3.10
- PyTorch 2.6.0+MetaX
- MACA toolkit at `/opt/maca`

## Build and run

```bash
# Build only
bash build.sh

# Run the public evaluator
bash run_bench.sh /path/to/auto_bench.py /path/to/v0.py
```

`AUTO_BENCH` and `V0_FILE` may be set instead of positional arguments.

## Performance

Public evaluator, 200 warmup iterations and 500 timed iterations:

| Implementation | Time (ms) | Speedup | Accuracy |
| --- | ---: | ---: | --- |
| Reference | 0.488472 | 1.000x | PASS |
| Optimized | 0.058832 | 8.303x | PASS |
