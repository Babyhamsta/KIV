"""
KIV: K-Indexed V Materialization for HuggingFace transformers.

Replaces the standard KV cache for global attention layers with a tiered
system that keeps K indexed for fast lookup and V on CPU for on-demand
retrieval. No model weights are modified. Works with any HF model that
uses DynamicCache.
"""

from .config import KIVConfig
from .cold_store import ColdKVStore
from .model_topology import ModelTopology, detect_topology
from .tiered_cache import TieredKVCache
from .middleware import KIVMiddleware

__all__ = [
    "KIVConfig",
    "ColdKVStore",
    "ModelTopology",
    "detect_topology",
    "TieredKVCache",
    "KIVMiddleware",
]
