import torch
import torch.nn as nn

try:
    import torch_npu
except ImportError:
    pass


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, pos, size, start=None, end=None):
        if pos.dim() != 2:
            raise ValueError(f"pos should be 2-dimensional, got {pos.dim()}-dimensional")
        if size.dim() != 1:
            raise ValueError(f"size should be 1-dimensional, got {size.dim()}-dimensional")
        if pos.size(1) != size.size(0):
            raise ValueError(f"Dimension mismatch: pos has {pos.size(1)} dimensions, but size has {size.size(0)} dimensions")

        N, D = pos.shape
        device = pos.device

        if start is None:
            start = torch.zeros(D, device=device)
        else:
            if start.dim() != 1 or start.size(0) != D:
                raise ValueError(f"start should have shape [{D}], got {start.shape}")

        if end is None:
            end = torch.max(pos, dim=0)[0] + size
        else:
            if end.dim() != 1 or end.size(0) != D:
                raise ValueError(f"end should have shape [{D}], got {end.shape}")

        grid_indices = ((pos - start.unsqueeze(0)) / size.unsqueeze(0)).long()
        grid_indices = torch.clamp(grid_indices, min=0)

        grid_counts = ((end - start) / size).long() + 1

        multipliers = torch.cumprod(
            torch.cat([torch.ones(1, device=device, dtype=torch.long), grid_counts[:-1]]), 
            dim=0
        )
        cluster_ids = (grid_indices * multipliers.unsqueeze(0)).sum(dim=1)

        unique_ids, inverse_indices = torch.unique(cluster_ids, return_inverse=True)
        return inverse_indices


def get_inputs():
    device = torch.device('npu') if hasattr(torch, 'npu') and torch.npu.is_available() else torch.device('cpu')
    pos = torch.tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]], device=device)
    size = torch.tensor([5, 5], device=device)
    end = torch.tensor([19, 19], device=device)
    return [pos, size, end]


def get_init_inputs():
    return []


if __name__ == "__main__":
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    print("Output:", output)
    print("Test passed on", inputs[0].device)
