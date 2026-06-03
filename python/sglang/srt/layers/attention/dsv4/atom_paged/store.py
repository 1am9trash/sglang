from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.deepseek_v4_rope import fused_norm_rope_inplace_triton


@triton.jit
def _scatter_rows_kernel(
    src_ptr,
    loc_ptr,
    valid_ptr,
    dst_ptr,
    dst_base,
    n_rows,
    D: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    if HAS_VALID:
        is_valid = tl.load(valid_ptr + row) != 0
        if not is_valid:
            return

    loc = tl.load(loc_ptr + row)
    if loc < 0:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(src_ptr + row * D + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + (dst_base + loc) * D + offs, vals, mask=mask)


def _scatter_rows(
    src: torch.Tensor,
    loc: torch.Tensor,
    dst: torch.Tensor,
    *,
    dst_base: int = 0,
    valid: Optional[torch.Tensor] = None,
) -> None:
    src = src.contiguous()
    loc = loc.to(torch.int32).contiguous()
    if valid is not None:
        valid = valid.to(torch.int32).contiguous()
    n_rows, head_dim = src.shape
    block_d = triton.next_power_of_2(head_dim)
    _scatter_rows_kernel[(n_rows,)](
        src,
        loc,
        valid if valid is not None else loc,
        dst,
        dst_base,
        n_rows,
        D=head_dim,
        HAS_VALID=valid is not None,
        BLOCK_D=block_d,
        num_warps=8,
    )


def store_swa_kv(
    *,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    unified_kv: torch.Tensor,
) -> None:
    kv = kv.contiguous()
    fused_norm_rope_inplace_triton(kv, kv_weight, eps, freqs_cis, positions=positions)
    _scatter_rows(kv, loc, unified_kv)


def store_compressed_kv(
    *,
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    unified_kv: torch.Tensor,
    atom_swa_slots: int,
    valid: Optional[torch.Tensor] = None,
) -> None:
    kv = kv.contiguous()
    fused_norm_rope_inplace_triton(kv, norm_weight, norm_eps, freqs_cis, positions=positions)
    _scatter_rows(kv, loc, unified_kv, dst_base=atom_swa_slots, valid=valid)


