"""Auto-detect model architecture from HuggingFace models."""

from __future__ import annotations

import logging
from typing import Any

from .model_topology import ModelTopology

logger = logging.getLogger(__name__)


def _resolve_text_config(model: Any) -> Any:
    """Unwrap multimodal wrappers to find the text config."""
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def _find_text_model(model: Any) -> Any:
    """Navigate HF model hierarchy to find the text model with layers."""
    if hasattr(model, "model"):
        inner = model.model
        # Gemma4 multimodal: model.model.language_model
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model
        # Gemma3n: model.model.text_model
        if hasattr(inner, "text_model") and hasattr(inner.text_model, "layers"):
            return inner.text_model
        # Llama, Mistral, etc.: model.model.layers
        if hasattr(inner, "layers"):
            return inner
    raise AttributeError(
        "Cannot find text model layers. "
        "Supported: model.model.language_model, model.model.text_model, model.model"
    )


def _detect_global_layers(config: Any) -> tuple[int, ...]:
    """Determine which layers use full (non-sliding) attention."""
    n_layers = config.num_hidden_layers

    # Pattern 1: explicit layer_types list (Gemma4/3/3n, Cohere2, Llama4)
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        global_indices = tuple(
            i for i, t in enumerate(layer_types) if t == "full_attention"
        )
        if global_indices:
            return global_indices
        # Llama4 uses "chunked_attention" / "full_attention"
        # If no "full_attention" found, treat all as global
        logger.warning(
            "layer_types present but no 'full_attention' layers found. "
            "Treating all %d layers as global.",
            n_layers,
        )
        return tuple(range(n_layers))

    # Pattern 2: uniform sliding_window (Mistral) — treat all as global for KIV
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is not None:
        logger.info(
            "Model has uniform sliding_window=%d. "
            "KIV will manage all %d layers as global.",
            sliding_window,
            n_layers,
        )
        return tuple(range(n_layers))

    # Pattern 3: no sliding window info — pure global attention (Llama, Phi-3)
    return tuple(range(n_layers))


def _detect_kv_sharing(
    config: Any, model: Any, global_indices: tuple[int, ...]
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Detect KV sharing among global layers.

    Returns (independent_kv_layers, kv_sharing_map).
    """
    n_layers = config.num_hidden_layers

    # Method 1: num_kv_shared_layers config field (Gemma4, Gemma3n)
    num_shared = getattr(config, "num_kv_shared_layers", 0) or 0
    if num_shared > 0:
        shared_start = n_layers - num_shared
        independent = tuple(i for i in global_indices if i < shared_start)
        shared_targets = tuple(i for i in global_indices if i >= shared_start)

        if not independent:
            # All global layers are in the shared range — no sharing possible
            return global_indices, {}

        source = independent[-1]
        sharing_map = {t: source for t in shared_targets}
        return independent, sharing_map

    # Method 2: probe attention modules for is_kv_shared_layer attribute
    try:
        text_model = _find_text_model(model)
        independent = []
        sharing_map = {}
        for layer_idx in global_indices:
            attn = text_model.layers[layer_idx].self_attn
            if getattr(attn, "is_kv_shared_layer", False):
                source_idx = getattr(attn, "kv_shared_layer_index", None)
                if source_idx is not None:
                    sharing_map[layer_idx] = source_idx
                else:
                    # Marked shared but no source — treat as independent
                    independent.append(layer_idx)
            else:
                independent.append(layer_idx)
        if sharing_map:
            return tuple(independent), sharing_map
    except (AttributeError, IndexError):
        pass

    # No sharing detected
    return global_indices, {}


def _detect_head_geometry(config: Any) -> tuple[int, int, int]:
    """Detect (num_query_heads, num_kv_heads, head_dim) for global layers."""
    num_query_heads = config.num_attention_heads

    # KV heads: prefer global-specific field (Gemma4), then standard
    num_kv_heads = getattr(config, "num_global_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_query_heads  # MHA

    # Head dim: prefer global_head_dim (Gemma4), then head_dim, then compute
    head_dim = getattr(config, "global_head_dim", None)
    if head_dim is None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // num_query_heads

    return num_query_heads, num_kv_heads, head_dim


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


def detect_topology(model: Any) -> ModelTopology:
    """Auto-detect model topology from a loaded HuggingFace model.

    Inspects model config, layer_types, and attention module attributes
    to build a ModelTopology describing the architecture.

    For unsupported models, falls back to treating all layers as global
    with no KV sharing. Logs a warning suggesting manual configuration.

    Args:
        model: A loaded HuggingFace PreTrainedModel.

    Returns:
        ModelTopology with detected architecture parameters.
    """
    config = _resolve_text_config(model)
    model_type = getattr(config, "model_type", "unknown")
    family = _MODEL_TYPE_MAP.get(model_type)

    if family is None:
        logger.warning(
            "Unknown model_type '%s'. Treating all layers as global with no "
            "KV sharing. For better efficiency, pass a manual ModelTopology.",
            model_type,
        )
        family = model_type

    # Detect layers
    global_indices = _detect_global_layers(config)
    independent_kv_layers, kv_sharing_map = _detect_kv_sharing(
        config, model, global_indices
    )

    # Detect heads
    num_query_heads, num_kv_heads, head_dim = _detect_head_geometry(config)

    topology = ModelTopology(
        model_family=family,
        num_hidden_layers=config.num_hidden_layers,
        global_layer_indices=global_indices,
        independent_kv_layers=independent_kv_layers,
        kv_sharing_map=kv_sharing_map,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    logger.info(
        "KIV topology detected for %s: %d global layers (%d independent, %d shared), "
        "%d query heads, %d KV heads, head_dim=%d",
        family,
        len(global_indices),
        len(independent_kv_layers),
        len(kv_sharing_map),
        num_query_heads,
        num_kv_heads,
        head_dim,
    )

    return topology
