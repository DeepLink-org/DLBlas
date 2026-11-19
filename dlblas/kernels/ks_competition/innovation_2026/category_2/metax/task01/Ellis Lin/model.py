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
        num_ngram_layers = multipliers.shape[0]
        max_ngram_size = multipliers.shape[1]
        prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)
        ans = [[] for _ in range(num_ngram_layers)]
        hashes = prod[:, :, 0].clone()
        for i in range(1, max_ngram_size):
            hashes.bitwise_xor_(prod[:, :, i])
            for layer_idx in range(num_ngram_layers):
                ans[layer_idx].append(
                    (
                        hashes[layer_idx].unsqueeze(-1)
                        % vocab_sizes[layer_idx, i - 1]
                        .to(torch.int64)
                        .unsqueeze(0)
                    ).to(torch.int32)
                )
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx] = torch.cat(ans[layer_idx], dim=-1)
        output = torch.stack(ans, dim=0)
        return output + offsets.unsqueeze(1)


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    offsets_list = []
    for layer_idx in range(vocab_sizes.shape[0]):
        flat = vocab_sizes[layer_idx].view(-1)
        prefix = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=flat.device),
                flat[:-1].cumsum(0, dtype=torch.int32),
            ]
        )
        offsets_list.append(prefix)
    return torch.stack(offsets_list, dim=0)


def generate_test_data(params):
    num_tokens = params["num_tokens"]
    max_ngram_size = params["ngram"]
    num_ngram_layers = params["layers"]
    num_embed_table_per_ngram = params["tables"]
    ngram_token_ids = torch.randint(
        0, 100000, (num_tokens, max_ngram_size), dtype=torch.int32
    )
    multipliers = torch.randint(
        0, 100000, (num_ngram_layers, max_ngram_size), dtype=torch.int64
    )
    vocab_sizes = torch.randint(
        100000,
        1000000,
        (
            num_ngram_layers,
            max_ngram_size - 1,
            num_embed_table_per_ngram,
        ),
        dtype=torch.int32,
    )
    offsets = make_offsets(vocab_sizes)
    return ngram_token_ids, multipliers, vocab_sizes, offsets


def get_inputs():
    return list(
        generate_test_data(
            {"num_tokens": 4096, "ngram": 3, "layers": 2, "tables": 8}
        )
    )


def get_init_inputs():
    return []
