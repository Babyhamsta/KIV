"""Two-partition attention with log-sum-exp merge."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .cold_store import ColdKVStore, _repeat_kv
from .config import KIVConfig


def two_partition_attention(
    query: torch.Tensor,
    hot_key: torch.Tensor,
    hot_value: torch.Tensor,
    cold_store: ColdKVStore,
    hot_mask: torch.Tensor | None,
    config: KIVConfig,
    scaling: float,
) -> torch.Tensor:
    """
    Attention over hot cache + top-P cold entries with log-sum-exp merge.

    Args:
        query:     [B, H_q, Q, D] — 8 query heads
        hot_key:   [B, H_kv, H_len, D] — 1 KV head, hot tokens
        hot_value: [B, H_kv, H_len, D]
        cold_store: ColdKVStore with K-index on VRAM, V on CPU
        hot_mask:  [B, 1, Q, H_len] causal mask (additive, -inf for invalid)
        config:    KIVConfig
        scaling:   head_dim ** -0.5

    Returns:
        [B, H_q, Q, D] attention output
    """
    num_kv_groups = config.num_query_heads // config.num_kv_heads

    # ── Partition 1: Hot (exact) ──
    hot_k = _repeat_kv(hot_key, num_kv_groups)    # [B, 8, H_len, D]
    hot_v = _repeat_kv(hot_value, num_kv_groups)   # [B, 8, H_len, D]

    S_hot = torch.matmul(query, hot_k.transpose(-2, -1)) * scaling  # [B, 8, Q, H_len]
    if hot_mask is not None:
        S_hot = S_hot + hot_mask

    # Numerics: compute max, exp, sum for hot partition
    m_hot = S_hot.max(dim=-1, keepdim=True).values        # [B, 8, Q, 1]
    exp_hot = torch.exp(S_hot - m_hot)                     # [B, 8, Q, H_len]
    l_hot = exp_hot.sum(dim=-1, keepdim=True)              # [B, 8, Q, 1]
    o_hot = torch.matmul(exp_hot, hot_v)                   # [B, 8, Q, D] (unnormalized)

    # ── Check cold partition ──
    cold_len = cold_store.cold_length
    if cold_len == 0:
        # No cold entries — just normalize hot and return
        return o_hot / l_hot

    # ── Partition 2: Cold (top-P approximate via page-based coarse-to-fine) ──

    top_scores, V_fetched = cold_store.score_and_fetch_cold(query, scaling, config)
    if top_scores is None:
        return o_hot / l_hot

    # Cold attention numerics
    m_cold = top_scores.max(dim=-1, keepdim=True).values   # [B, 8, Q, 1]
    exp_cold = torch.exp(top_scores - m_cold)               # [B, 8, Q, P]
    l_cold = exp_cold.sum(dim=-1, keepdim=True)             # [B, 8, Q, 1]
    # exp_cold: [B, 8, Q, P] x V_fetched: [B, 8, Q, P, D] -> [B, 8, Q, D]
    o_cold = torch.einsum("bhqp,bhqpd->bhqd", exp_cold, V_fetched)

    # ── Log-sum-exp merge ──
    m_new = torch.maximum(m_hot, m_cold)                   # [B, 8, Q, 1]
    corr_hot = torch.exp(m_hot - m_new)                     # [B, 8, Q, 1]
    corr_cold = torch.exp(m_cold - m_new)                   # [B, 8, Q, 1]
    l_new = corr_hot * l_hot + corr_cold * l_cold           # [B, 8, Q, 1]
    output = (corr_hot * o_hot + corr_cold * o_cold) / l_new  # [B, 8, Q, D]

    return output
