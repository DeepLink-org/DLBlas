import time
import sys
sys.path.insert(0, '/mnt/opt_test/engram_hash_run')
from engram_hash_ref import Model, make_offsets, generate_test_data, get_inputs

import torch

def main():
    print(f"Torch version: {torch.__version__}")
    
    # Use the same input generation as the MACA test
    # Test data with seed=42 for reproducibility
    import numpy as np
    rng = np.random.RandomState(42)
    
    num_tokens = 4096
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8
    
    ngram_token_ids = torch.from_numpy(rng.randint(0, 100000, (num_tokens, max_ngram_size)).astype(np.int32))
    multipliers = torch.from_numpy(rng.randint(0, 100000, (num_ngram_layers, max_ngram_size)).astype(np.int64))
    vocab_sizes = torch.from_numpy(rng.randint(100000, 1000000, (num_ngram_layers, max_ngram_size-1, num_embed_table_per_ngram)).astype(np.int32))
    offsets = make_offsets(vocab_sizes)
    
    model = Model()
    
    # Warmup
    warmup = 10
    for _ in range(warmup):
        _ = model.forward(ngram_token_ids, multipliers, vocab_sizes, offsets)
    
    # Benchmark
    n_iter = 100
    start = time.perf_counter()
    for _ in range(n_iter):
        result = model.forward(ngram_token_ids, multipliers, vocab_sizes, offsets)
    end = time.perf_counter()
    avg_time_ms = (end - start) / n_iter * 1000.0
    
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Torch CPU average time: {avg_time_ms:.6f} ms")
    
    print(f"\n=== PERFORMANCE COMPARISON ===")
    print(f"Torch (CPU) time:  {avg_time_ms:.6f} ms")
    print(f"MACA C500 time:    0.019377 ms")
    print(f"MACA speedup:      {avg_time_ms / 0.019377:.2f}x vs Torch CPU")
    
    # Save result
    with open('/mnt/opt_test/engram_hash_run/torch_comparison.txt', 'w') as f:
        f.write(f"Torch version: {torch.__version__}\n")
        f.write(f"Torch (CPU) time: {avg_time_ms:.6f} ms\n")
        f.write(f"MACA C500 time: 0.019377 ms\n")
        f.write(f"MACA speedup: {avg_time_ms / 0.019377:.2f}x vs Torch CPU\n")
        f.write(f"Output shape: {result.shape}\n")
        f.write(f"Parameters: num_tokens={num_tokens}, max_ngram_size={max_ngram_size}, num_ngram_layers={num_ngram_layers}, num_embed_table_per_ngram={num_embed_table_per_ngram}\n")
    
    print("Comparison saved to torch_comparison.txt")
    
if __name__ == '__main__':
    main()
