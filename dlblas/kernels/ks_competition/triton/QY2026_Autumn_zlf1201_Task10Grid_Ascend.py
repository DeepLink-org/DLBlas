# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _cluster_ids_2d_kernel(pos_ptr, size_ptr, end_ptr, out_ptr, N,
                               BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        s0 = tl.load(size_ptr + 0)
        s1 = tl.load(size_ptr + 1)
        e1 = tl.load(end_ptr + 1)

        inv_s0 = 1.0 / s0
        inv_s1 = 1.0 / s1

        grid_count_1 = tl.cast(e1 * inv_s1, tl.int64) + 1

        base = offs * 2

        p0 = tl.load(pos_ptr + base + 0, mask=mask, other=0.0)
        p1 = tl.load(pos_ptr + base + 1, mask=mask, other=0.0)

        r0 = tl.maximum(p0 * inv_s0, 0.0)
        r1 = tl.maximum(p1 * inv_s1, 0.0)

        idx0 = tl.cast(r0, tl.int64)
        idx1 = tl.cast(r1, tl.int64)

        cluster_ids = idx0 * grid_count_1 + idx1

        tl.store(out_ptr + offs, cluster_ids, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def _inverse_from_cluster_ids(self, cluster_ids):
        order = torch.argsort(cluster_ids.to(torch.float32))

        sorted_ids = cluster_ids[order]

        flags = torch.ones(
            sorted_ids.shape,
            dtype=torch.bool,
            device=cluster_ids.device
        )
        flags[1:] = sorted_ids[1:] != sorted_ids[:-1]

        ranks_i32 = torch.cumsum(flags.to(torch.int32), dim=0) - 1
        ranks = ranks_i32.to(torch.long)

        inverse_indices = torch.empty(
            cluster_ids.shape,
            dtype=torch.long,
            device=cluster_ids.device
        )
        inverse_indices[order] = ranks

        return inverse_indices

    def forward(self, pos, size, start=None, end=None):
        if pos.dim() != 2:
            raise ValueError(f"pos should be 2-dimensional, got {pos.dim()}-dimensional")

        if size.dim() != 1:
            raise ValueError(f"size should be 1-dimensional, got {size.dim()}-dimensional")

        if pos.size(1) != size.size(0):
            raise ValueError(
                f"Dimension mismatch: pos has {pos.size(1)} dimensions, "
                f"but size has {size.size(0)} dimensions"
            )

        N, D = pos.shape
        device = pos.device

        pos_calc = pos if pos.dtype == torch.float32 else pos.to(dtype=torch.float32)

        size_calc = size
        if size_calc.dtype != torch.float32 or size_calc.device != device:
            size_calc = size.to(device=device, dtype=torch.float32)

        if D == 2 and start is None and end is not None:
            end_calc = end
            if end_calc.dtype != torch.float32 or end_calc.device != device:
                end_calc = end.to(device=device, dtype=torch.float32)

            if N < 2048:
                grid_count_1 = (end_calc[1] / size_calc[1]).long() + 1

                idx0 = (pos_calc[:, 0] / size_calc[0]).long().clamp_min(0)
                idx1 = (pos_calc[:, 1] / size_calc[1]).long().clamp_min(0)

                cluster_ids = idx0 * grid_count_1 + idx1

                return self._inverse_from_cluster_ids(cluster_ids)

            if _TRITON_AVAILABLE and device.type in ("cuda", "npu"):
                try:
                    pos_ctg = pos_calc.contiguous()
                    size_ctg = size_calc.contiguous()
                    end_ctg = end_calc.contiguous()

                    cluster_ids = torch.empty((N,), dtype=torch.long, device=device)

                    block_size = 512
                    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

                    _cluster_ids_2d_kernel[grid](
                        pos_ctg,
                        size_ctg,
                        end_ctg,
                        cluster_ids,
                        N,
                        BLOCK_SIZE=block_size,
                        num_warps=8,
                        num_stages=1,
                        fast_math=True,
                    )

                    return self._inverse_from_cluster_ids(cluster_ids)

                except Exception:
                    pass

            grid_count_1 = (end_calc[1] / size_calc[1]).long() + 1

            idx0 = (pos_calc[:, 0] / size_calc[0]).long().clamp_min(0)
            idx1 = (pos_calc[:, 1] / size_calc[1]).long().clamp_min(0)

            cluster_ids = idx0 * grid_count_1 + idx1

            return self._inverse_from_cluster_ids(cluster_ids)

        if start is None:
            start_calc = torch.zeros(D, device=device, dtype=torch.float32)
        else:
            if start.dim() != 1 or start.size(0) != D:
                raise ValueError(f"start should have shape [{D}], got {start.shape}")

            start_calc = start
            if start_calc.dtype != torch.float32 or start_calc.device != device:
                start_calc = start.to(device=device, dtype=torch.float32)

        if end is None:
            end_calc = torch.max(pos_calc, dim=0)[0] + size_calc
        else:
            if end.dim() != 1 or end.size(0) != D:
                raise ValueError(f"end should have shape [{D}], got {end.shape}")

            end_calc = end
            if end_calc.dtype != torch.float32 or end_calc.device != device:
                end_calc = end.to(device=device, dtype=torch.float32)

        grid_counts = ((end_calc - start_calc) / size_calc).long() + 1

        if D == 1:
            idx0 = ((pos_calc[:, 0] - start_calc[0]) / size_calc[0]).long().clamp_min(0)
            cluster_ids = idx0

        elif D == 2:
            idx0 = ((pos_calc[:, 0] - start_calc[0]) / size_calc[0]).long().clamp_min(0)
            idx1 = ((pos_calc[:, 1] - start_calc[1]) / size_calc[1]).long().clamp_min(0)
            cluster_ids = idx0 * grid_counts[1] + idx1

        elif D == 3:
            idx0 = ((pos_calc[:, 0] - start_calc[0]) / size_calc[0]).long().clamp_min(0)
            idx1 = ((pos_calc[:, 1] - start_calc[1]) / size_calc[1]).long().clamp_min(0)
            idx2 = ((pos_calc[:, 2] - start_calc[2]) / size_calc[2]).long().clamp_min(0)

            cluster_ids = (idx0 * grid_counts[1] + idx1) * grid_counts[2] + idx2

        else:
            idx0 = ((pos_calc[:, 0] - start_calc[0]) / size_calc[0]).long().clamp_min(0)
            cluster_ids = idx0

            for d in range(1, D):
                idxd = ((pos_calc[:, d] - start_calc[d]) / size_calc[d]).long().clamp_min(0)
                cluster_ids = cluster_ids * grid_counts[d] + idxd

        return self._inverse_from_cluster_ids(cluster_ids)