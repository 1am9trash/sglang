from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _build_hca_decode_indices_kernel(
    swa_indices_ptr,
    swa_lengths_ptr,
    c128_indices_ptr,
    c128_lengths_ptr,
    out_indices_ptr,
    out_indptr_ptr,
    atom_swa_slots,
    swa_width: tl.constexpr,
    c128_width: tl.constexpr,
    total_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = block_id * BLOCK + tl.arange(0, BLOCK)

    if block_id == 0:
        tl.store(out_indptr_ptr + t, t * total_width)
        if t == tl.num_programs(0) - 1:
            tl.store(out_indptr_ptr + t + 1, (t + 1) * total_width)

    swa_len = tl.load(swa_lengths_ptr + t)
    c128_len = tl.load(c128_lengths_ptr + t)

    swa_mask = offs < swa_width
    swa_vals = tl.load(
        swa_indices_ptr + t * swa_width + offs,
        mask=swa_mask,
        other=-1,
    )
    swa_valid = swa_mask & (offs < swa_len) & (swa_vals >= 0)
    tl.store(
        out_indices_ptr + t * total_width + offs,
        tl.where(swa_valid, swa_vals, -1),
        mask=swa_mask,
    )

    c_offs = offs - swa_width
    c_mask = (offs >= swa_width) & (c_offs < c128_width)
    c_vals = tl.load(
        c128_indices_ptr + t * c128_width + c_offs,
        mask=c_mask,
        other=-1,
    )
    c_valid = c_mask & (c_offs < c128_len) & (c_vals >= 0)
    tl.store(
        out_indices_ptr + t * total_width + offs,
        tl.where(c_valid, atom_swa_slots + c_vals, -1),
        mask=c_mask,
    )


def build_hca_decode_indices(
    *,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    c128_indices: torch.Tensor,
    c128_lengths: torch.Tensor,
    atom_swa_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ATOM-style flat HCA decode indices from SGLang padded metadata.

    This first experimental path keeps a fixed per-token width and uses -1
    padding. The ATOM paged decode kernel already skips -1 entries.
    """
    if swa_indices.dim() == 3:
        assert swa_indices.shape[1] == 1
        swa_indices = swa_indices[:, 0, :]
    if c128_indices.dim() == 3:
        assert c128_indices.shape[1] == 1
        c128_indices = c128_indices[:, 0, :]

    swa_indices = swa_indices.contiguous()
    c128_indices = c128_indices.contiguous()
    swa_lengths = swa_lengths.to(torch.int32).contiguous()
    c128_lengths = c128_lengths.to(torch.int32).contiguous()

    T, swa_width = swa_indices.shape
    assert c128_indices.shape[0] == T
    c128_width = c128_indices.shape[1]
    total_width = swa_width + c128_width
    out_indices = torch.empty((T, total_width), dtype=torch.int32, device=swa_indices.device)
    out_indptr = torch.empty((T + 1,), dtype=torch.int32, device=swa_indices.device)
    block = min(1024, triton.next_power_of_2(total_width))
    grid = (T, triton.cdiv(total_width, block))

    _build_hca_decode_indices_kernel[grid](
        swa_indices,
        swa_lengths,
        c128_indices,
        c128_lengths,
        out_indices,
        out_indptr,
        atom_swa_slots,
        swa_width,
        c128_width,
        total_width,
        BLOCK=block,
        num_warps=8,
    )
    return out_indices.view(-1), out_indptr

