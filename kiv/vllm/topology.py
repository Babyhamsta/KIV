"""Detect model topology from a vLLM model config."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..model_topology import ModelTopology

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


# Known model_type values mapped to family names
_MODEL_TYPE_MAP = {
    "gemma4_text": "gemma4",
    "gemma4": "gemma4",
    "gemma3_text": "gemma3",
    "gemma3": "gemma3",
    "gemma3n_text": "gemma3n",
    "gemma3n": "gemma3n",
    "gemma2": "gemma2",
    "gemma": "gemma",
    "llama": "llama",
    "mistral": "mistral",
    "cohere2": "cohere2",
    "cohere": "cohere",
    "phi3": "phi3",
    "phi": "phi",
    "qwen2": "qwen2",
}


def _detect_global_layers(hf_config: Any) -> tuple[int, ...]:
    """Determine which layers use full (non-sliding) attention."""
    n_layers = hf_config.num_hidden_layers

    # Pattern 1: explicit layer_types list
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None:
        global_indices = tuple(
            i for i, t in enumerate(layer_types) if t == "full_attention"
        )
        if global_indices:
            return global_indices
        logger.warning(
            "layer_types present but no 'full_attention' layers found. "
            "Treating all %d layers as global.",
            n_layers,
        )
        return tuple(range(n_layers))

    # Pattern 2: uniform sliding_window — treat all as global for KIV
    sliding_window = getattr(hf_config, "sliding_window", None)
    if sliding_window is not None:
        logger.info(
            "Model has uniform sliding_window=%d. "
            "KIV will manage all %d layers as global.",
            sliding_window,
            n_layers,
        )
        return tuple(range(n_layers))

    # Pattern 3: pure global attention
    return tuple(range(n_layers))


def _detect_kv_sharing(
    hf_config: Any, global_indices: tuple[int, ...]
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Detect KV sharing among global layers from config fields only.

    Unlike the HF version, this cannot probe model modules — vLLM models
    are not loaded as HF PreTrainedModels. We rely on config fields only.
    """
    n_layers = hf_config.num_hidden_layers

    # num_kv_shared_layers config field (Gemma4, Gemma3n)
    num_shared = getattr(hf_config, "num_kv_shared_layers", 0) or 0
    if num_shared > 0:
        shared_start = n_layers - num_shared
        independent = tuple(i for i in global_indices if i < shared_start)
        shared_targets = tuple(i for i in global_indices if i >= shared_start)

        if not independent:
            return global_indices, {}

        source = independent[-1]
        sharing_map = {t: source for t in shared_targets}
        return independent, sharing_map

    # No sharing detected from config
    return global_indices, {}


def _detect_head_geometry(hf_config: Any) -> tuple[int, int, int]:
    """Detect (num_query_heads, num_kv_heads, head_dim) for global layers."""
    num_query_heads = hf_config.num_attention_heads

    num_kv_heads = getattr(hf_config, "num_global_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_query_heads

    head_dim = getattr(hf_config, "global_head_dim", None)
    if head_dim is None:
        head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        head_dim = hf_config.hidden_size // num_query_heads

    return num_query_heads, num_kv_heads, head_dim


def detect_topology_from_vllm(vllm_config: VllmConfig) -> ModelTopology:
    """Auto-detect model topology from a vLLM config object.

    Args:
        vllm_config: A ``VllmConfig`` instance. The HuggingFace model
            config is accessed via ``vllm_config.model_config.hf_config``.

    Returns:
        ModelTopology with detected architecture parameters.
    """
    hf_config = vllm_config.model_config.hf_config

    # Unwrap text_config if present (multimodal models)
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    model_type = getattr(hf_config, "model_type", "unknown")
    family = _MODEL_TYPE_MAP.get(model_type, model_type)

    if model_type not in _MODEL_TYPE_MAP:
        logger.warning(
            "Unknown model_type '%s'. Treating all layers as global with no "
            "KV sharing. For better efficiency, pass a manual ModelTopology.",
            model_type,
        )

    global_indices = _detect_global_layers(hf_config)
    independent_kv_layers, kv_sharing_map = _detect_kv_sharing(
        hf_config, global_indices
    )
    num_query_heads, num_kv_heads, head_dim = _detect_head_geometry(hf_config)

    topology = ModelTopology(
        model_family=family,
        num_hidden_layers=hf_config.num_hidden_layers,
        global_layer_indices=global_indices,
        independent_kv_layers=independent_kv_layers,
        kv_sharing_map=kv_sharing_map,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    logger.info(
        "KIV topology detected for %s (vLLM): %d global layers "
        "(%d independent, %d shared), %d query heads, %d KV heads, "
        "head_dim=%d",
        family,
        len(global_indices),
        len(independent_kv_layers),
        len(kv_sharing_map),
        num_query_heads,
        num_kv_heads,
        head_dim,
    )

    return topology
