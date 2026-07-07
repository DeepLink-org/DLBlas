import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(
        self,
        ngram_token_ids: torch.Tensor,
        multipliers: torch.Tensor,
        vocab_sizes: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Pure PyTorch reference implementation of engram hash.
        Args:
            ngram_token_ids: N-gram token IDs of shape (num_tokens, max_ngram_size), int32.
            multipliers: Per-layer hash multipliers of shape (num_ngram_layers, max_ngram_size), int64.
            vocab_sizes: Per-layer per-ngram embedding table sizes of shape
                (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
            offsets: Per-layer embedding table offsets of shape
                (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
        Returns:
            Embedding indices of shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
        """
        num_ngram_layers = multipliers.shape[0]
        max_ngram_size = multipliers.shape[1]
        prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)
        ans = [[] for _ in range(num_ngram_layers)]
        hashes = prod[:, :, 0].clone()
        for i in range(1, max_ngram_size):
            hashes.bitwise_xor_(prod[:, :, i])
            for layer_idx in range(num_ngram_layers):
                ans[layer_idx].append((hashes[layer_idx].unsqueeze(-1) % vocab_sizes[layer_idx, i - 1].to(torch.int64).unsqueeze(0)).to(torch.int32))
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)
        output = torch.stack(ans, dim=0)
        return output + offsets.unsqueeze(1)
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