"""
KIV: K-Indexed V Materialization for Gemma 4 E2B.

Replaces the standard KV cache for global attention layers with a tiered
system that keeps K indexed for fast lookup and V on CPU for on-demand
retrieval. No model weights are modified.
"""

from .config import KIVConfig
from .cold_store import ColdKVStore
from .tiered_cache import TieredKVCache
from .partitioned_attn import two_partition_attention
from .middleware import KIVMiddleware

__all__ = [
    "KIVConfig",
    "ColdKVStore",
    "TieredKVCache",
    "two_partition_attention",
    "KIVMiddleware",
]
