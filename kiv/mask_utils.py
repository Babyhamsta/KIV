"""Attention mask utilities for KIV."""

from __future__ import annotations

import torch


def extend_mask_for_cold(
    mask: torch.Tensor, num_cold: int
) -> torch.Tensor:
    """Prepend columns to attention mask for cold tokens (all attendable).

    Handles multiple mask conventions:
      - Float masks (additive): 0.0 = attend, -inf = ignore -> prepend 0.0
      - Bool masks: True = attend -> prepend True
      - 2D masks [B, KV] or 4D masks [B, H, Q, KV]
    """
    if mask.dtype == torch.bool:
        fill_value = True
    else:
        fill_value = 0.0

    if mask.ndim == 2:
        # [B, KV] -> prepend along last dim
        cold_cols = mask.new_full((mask.shape[0], num_cold), fill_value)
    elif mask.ndim == 4:
        # [B, H, Q, KV] -> prepend along last dim
        cold_cols = mask.new_full(
            (mask.shape[0], mask.shape[1], mask.shape[2], num_cold),
            fill_value,
        )
    else:
        # Unknown shape — try prepending on last dim
        shape = list(mask.shape)
        shape[-1] = num_cold
        cold_cols = mask.new_full(shape, fill_value)

    return torch.cat([cold_cols, mask], dim=-1)
