"""Vendored from ATOM atom/model_ops/sparse_attn_v4.py — pure-torch sparse-attn
reference (used only by *_reference paths for correctness bisection). No atom.* dep."""
from __future__ import annotations
import torch

def _sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention with per-query top-k KV gather and per-head sink.

    Reference: /data/DeepSeek-V4-Pro/inference/kernel.py:276-368

    For each query position (b, m), gathers the top-k KV positions selected by
    `topk_idxs[b, m, :]` (with -1 entries skipped), computes scaled dot-product
    attention with all H heads sharing the single-headed `kv`, and includes a
    per-head learnable `attn_sink` logit in the softmax denominator only.

    Args:
        q:           [B, M, H, D]  query, BF16
        kv:          [B, N, D]     shared key=value, single head (MQA), BF16
        attn_sink:   [H,]          per-head sink logit, FP32
        topk_idxs:   [B, M, K]     selected KV positions, INT32. -1 = skip.
        softmax_scale: scalar      softmax scale (typically D ** -0.5)

    Returns:
        o: [B, M, H, D] BF16

    Notes:
        - The sink contributes only to the denominator; it never appears as
          attention weight on a KV position. Letting `attn_sink[h] = -inf`
          recovers standard sparse attention without sink.
        - Internal accumulation is FP32. Output is cast back to q.dtype.
        - Invalid (-1) topk entries set their logit to -inf, contributing 0 to
          softmax and producing zero contribution to the output.
        - When all K entries are invalid for some (b, m, h), the result is 0
          (sum_exp = exp(sink - (-inf)) = 0; division below uses safe eps).
    """
    B, M, H, D = q.shape
    _, N, D_kv = kv.shape
    K = topk_idxs.shape[-1]
    assert kv.shape[0] == B, f"batch mismatch: q={B} vs kv={kv.shape[0]}"
    assert D_kv == D, f"head_dim mismatch: q={D} vs kv={D_kv}"
    assert attn_sink.shape == (H,), f"attn_sink shape {attn_sink.shape} != ({H},)"
    assert topk_idxs.shape == (B, M, K)

    out_dtype = q.dtype
    device = q.device

    # ----- Gather KV per query position -----
    # safe_idxs avoids out-of-bounds for the -1 sentinel; we mask the result below.
    valid = topk_idxs != -1  # [B, M, K] bool
    safe_idxs = topk_idxs.clamp(min=0).long()  # [B, M, K] int64

    # Advanced indexing: kv_gathered[b, m, k, :] = kv[b, safe_idxs[b, m, k], :]
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K)
    kv_gathered = kv[batch_idx, safe_idxs]  # [B, M, K, D]

    # Promote to FP32 for accumulation; zero out invalid positions in value tensor
    # so they contribute nothing to weighted sum even before masking the logits.
    kv_f32 = kv_gathered.float()
    kv_f32 = torch.where(
        valid.unsqueeze(-1), kv_f32, torch.zeros((), dtype=kv_f32.dtype, device=device)
    )

    # ----- Scores: q @ kv^T -----
    # q: [B, M, H, D]  ;  kv_f32: [B, M, K, D]  ->  scores: [B, M, H, K]
    q_f32 = q.float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q_f32, kv_f32) * float(softmax_scale)
    # Mask invalid positions in logits with -inf so they contribute 0 weight.
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # ----- Softmax with sink in denominator -----
    # Concat sink logit at the end: combined[..., :K] are real positions,
    # combined[..., K] is the per-head sink. Take softmax over (K+1), then drop
    # the sink column from the weights — its contribution stays in the
    # denominator via softmax normalization.
    sink = attn_sink.float().view(1, 1, H, 1).expand(B, M, H, 1)
    combined = torch.cat([scores, sink], dim=-1)  # [B, M, H, K+1]

    # Numerically stable softmax: subtract max along K+1 axis.
    # When all entries (including sink) are -inf for some (b,m,h), softmax of
    # all -inf is undefined; we get NaN. Replace with 0 in that pathological
    # case (matches kernel's behavior since `acc_o` stays 0 in that case).
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"),
        torch.zeros((), dtype=cmax.dtype, device=device),
        cmax,
    )
    weights = (combined - cmax).exp()
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom.clamp(min=1e-30)
    weights_kv = weights[..., :K]  # drop sink contribution from output side

    # ----- Weighted sum -----
    # weights_kv: [B, M, H, K]  ;  kv_f32: [B, M, K, D]  ->  out: [B, M, H, D]
    out = torch.einsum("bmhk,bmkd->bmhd", weights_kv, kv_f32)
    return out.to(out_dtype)



def _sparse_attn_ragged_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return _sparse_attn_torch(
        q.unsqueeze(0),
        kv.unsqueeze(0),
        attn_sink,
        topk_idxs.unsqueeze(0),
        softmax_scale,
    ).squeeze(0)


