"""Algorithm and budget configuration for KIV."""

from dataclasses import dataclass


@dataclass
class KIVConfig:
    """Algorithm and budget parameters for K-Indexed V Materialization.

    Model topology (layer indices, head counts, etc.) is detected
    automatically via ``detect_topology`` and stored in ``ModelTopology``.
    This dataclass holds only the tunable algorithm parameters.
    """

    # Hot cache: last N tokens keep exact K+V in VRAM
    hot_budget: int = 2048

    # Top-P: number of cold entries retrieved per decode step
    top_p: int = 256

    # Page-based coarse-to-fine retrieval
    page_size: int = 128         # tokens per page in cold store
    top_pages: int = 32          # pages selected in coarse pass

    # Performance
    prefetch_stream: bool = True  # async CUDA stream for V fetch
