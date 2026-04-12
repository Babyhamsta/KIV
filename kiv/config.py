"""Configuration for K-Indexed V Materialization."""

from dataclasses import dataclass


@dataclass
class KIVConfig:
    """Configuration for the tiered KV cache system."""

    # Hot cache: last N tokens keep exact K+V in VRAM
    hot_budget: int = 2048

    # Top-P: number of cold entries retrieved per decode step
    top_p: int = 256

    # Gemma 4 E2B layer topology
    global_layer_indices: tuple[int, ...] = (4, 9, 14, 19, 24, 29, 34)
    independent_kv_layers: tuple[int, ...] = (4, 9, 14)
    kv_shared_source: int = 14  # layer that sources shared KV for 19-34

    # Gemma 4 E2B attention dimensions (global layers)
    num_kv_heads: int = 1       # MQA: 1 KV head
    head_dim: int = 512          # global_head_dim
    num_query_heads: int = 8

    # Page-based coarse-to-fine retrieval
    page_size: int = 128         # tokens per page in cold store
    top_pages: int = 32          # pages selected in coarse pass

    # Performance
    prefetch_stream: bool = True  # async CUDA stream for V fetch
