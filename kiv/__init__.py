"""KIV: K-Indexed V Materialization."""

__version__ = "0.2.0"

from .config import KIVConfig
from .cold_store import ColdKVStore
from .model_topology import ModelTopology

__all__ = [
    "KIVConfig",
    "ColdKVStore",
    "ModelTopology",
    "detect_topology",
    "TieredKVCache",
    "KIVMiddleware",
]


def __getattr__(name: str):
    if name == "detect_topology":
        from .hf_topology import detect_topology

        globals()["detect_topology"] = detect_topology
        return detect_topology
    if name == "TieredKVCache":
        from .tiered_cache import TieredKVCache

        globals()["TieredKVCache"] = TieredKVCache
        return TieredKVCache
    if name == "KIVMiddleware":
        from .middleware import KIVMiddleware

        globals()["KIVMiddleware"] = KIVMiddleware
        return KIVMiddleware
    raise AttributeError(f"module 'kiv' has no attribute {name!r}")
