"""
KIV Middleware: installs tiered KV cache into any HuggingFace transformer
without modifying model weights or attention forwards.

Uses two model-agnostic hooks:
  1. TieredKVCache (extends DynamicCache) — handles hot storage + cold eviction
  2. Custom attention function — retrieves cold K/V, concatenates, delegates
     to standard attention (sdpa/flash/eager)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .config import KIVConfig
from .model_topology import ModelTopology, detect_topology, _find_text_model
from .tiered_cache import TieredKVCache

logger = logging.getLogger(__name__)


# ── Mask extension helper ──


def _extend_mask_for_cold(
    mask: torch.Tensor, num_cold: int
) -> torch.Tensor:
    """Prepend columns to attention mask for cold tokens (all attendable).

    Handles multiple mask conventions:
      - Float masks (additive): 0.0 = attend, -inf = ignore → prepend 0.0
      - Bool masks: True = attend → prepend True
      - 2D masks [B, KV] or 4D masks [B, H, Q, KV]
    """
    if mask.dtype == torch.bool:
        fill_value = True
    else:
        fill_value = 0.0

    if mask.ndim == 2:
        # [B, KV] → prepend along last dim
        cold_cols = mask.new_full((mask.shape[0], num_cold), fill_value)
    elif mask.ndim == 4:
        # [B, H, Q, KV] → prepend along last dim
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


class KIVMiddleware:
    """
    Installs K-Indexed V Materialization into any HuggingFace transformer.

    Does not modify model weights or attention forward functions. Instead:
    1. Registers a custom attention function that augments hot K/V with
       cold retrievals before delegating to standard attention.
    2. Provides TieredKVCache as a drop-in for DynamicCache.

    Usage:
        middleware = KIVMiddleware(model, config)
        middleware.install()
        cache = middleware.create_cache()
        output = model.generate(..., past_key_values=cache)
        middleware.uninstall()
    """

    def __init__(
        self,
        model: Any,
        config: KIVConfig | None = None,
        topology: ModelTopology | None = None,
    ) -> None:
        self.model = model
        self.config = config or KIVConfig()
        self._topology_override = topology
        self.topology: ModelTopology | None = None
        self._text_model: Any = None
        self._installed = False
        self._original_impl: str | None = None
        self._kiv_key: str | None = None

    def install(self) -> None:
        """Install KIV: detect topology, register attention function."""
        if self._installed:
            logger.warning("KIV middleware already installed.")
            return

        # Detect or use provided topology
        if self._topology_override is not None:
            self.topology = self._topology_override
            logger.info("Using manually provided ModelTopology.")
        else:
            self.topology = detect_topology(self.model)

        self._text_model = _find_text_model(self.model)

        # Mark global layers with their index
        for layer_idx in self.topology.global_layer_indices:
            attn = self._text_model.layers[layer_idx].self_attn
            attn._kiv_layer_idx = layer_idx

        # Register custom attention function (in both attention and mask registries)
        self._original_impl = self._resolve_attn_impl()
        original_fn = self._get_attn_fn(self._original_impl)
        self._kiv_key = f"kiv_{self._original_impl}"
        ALL_ATTENTION_FUNCTIONS[self._kiv_key] = self._make_kiv_attention(
            original_fn
        )
        # Mirror the mask function from the original implementation.
        # Must use _global_mapping because _preprocess_mask_arguments
        # checks _global_mapping directly (not the local override dict).
        if self._original_impl in ALL_MASK_ATTENTION_FUNCTIONS:
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[self._kiv_key] = (
                ALL_MASK_ATTENTION_FUNCTIONS[self._original_impl]
            )
        self._set_attn_impl(self._kiv_key)

        self._installed = True
        logger.info(
            "KIV middleware installed: %d global layers (%d independent, %d shared), "
            "attn=%s, hot=%d, top_p=%d",
            len(self.topology.global_layer_indices),
            len(self.topology.independent_kv_layers),
            len(self.topology.kv_sharing_map),
            self._original_impl,
            self.config.hot_budget,
            self.config.top_p,
        )

    def uninstall(self) -> None:
        """Remove KIV: restore original attention, clean up attributes."""
        if not self._installed:
            return

        # Restore original attention implementation
        if self._original_impl is not None:
            self._set_attn_impl(self._original_impl)

        # Remove registered attention and mask functions
        if self._kiv_key is not None:
            if self._kiv_key in ALL_ATTENTION_FUNCTIONS:
                del ALL_ATTENTION_FUNCTIONS[self._kiv_key]
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(self._kiv_key, None)

        # Clean up attributes on attention modules
        if self._text_model is not None and self.topology is not None:
            for layer_idx in self.topology.global_layer_indices:
                attn = self._text_model.layers[layer_idx].self_attn
                for attr in ("_kiv_layer_idx", "_kiv_cache"):
                    if hasattr(attn, attr):
                        delattr(attn, attr)

        self._installed = False
        self._original_impl = None
        self._kiv_key = None
        logger.info("KIV middleware uninstalled.")

    def create_cache(self, device: torch.device | None = None) -> TieredKVCache:
        """Create a TieredKVCache configured for this model."""
        if self.topology is None:
            raise RuntimeError("Call install() before create_cache().")

        if device is None:
            device = next(self.model.parameters()).device

        cache = TieredKVCache(
            config=self.model.config,
            kiv_config=self.config,
            topology=self.topology,
            device=device,
        )

        # Store cache reference on each global attention module
        for layer_idx in self.topology.global_layer_indices:
            attn = self._text_model.layers[layer_idx].self_attn
            attn._kiv_cache = cache

        return cache

    def chunked_prefill(
        self,
        input_ids: torch.Tensor,
        cache: TieredKVCache,
        chunk_size: int = 4096,
        prefill_hot_cap: int | None = None,
    ) -> torch.Tensor:
        """
        Process a long prompt in chunks to avoid quadratic attention OOM.

        Args:
            input_ids: [B, T] full prompt token IDs on GPU
            cache: TieredKVCache (should be fresh, pre-prefill)
            chunk_size: tokens per chunk (default 4096)
            prefill_hot_cap: if set, cap the hot cache at this many tokens
                during prefill. Oldest tokens evict to cold but cold is NOT
                attended during prefill (evict-only, no retrieval).
                This bounds VRAM at the cost of losing exact attention to
                early tokens. If None, hot cache grows unbounded.

        Returns:
            logits for the last token: [B, vocab_size]
        """
        B, T = input_ids.shape
        num_chunks = (T + chunk_size - 1) // chunk_size
        logits = None

        # During bounded prefill, suppress cold retrieval.
        # Evicted tokens go to cold but aren't re-attended — hot-only attention.
        if prefill_hot_cap is not None:
            cache._suppress_cold = True

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, T)
            chunk_ids = input_ids[:, start:end]

            with torch.no_grad():
                outputs = self.model(
                    input_ids=chunk_ids,
                    past_key_values=cache,
                    use_cache=True,
                )

            # Only keep logits from the last chunk
            if end == T:
                logits = outputs.logits[:, -1, :]

            # Free intermediate logits
            del outputs

            # Bounded prefill: evict excess from hot to cold (no re-attending)
            if prefill_hot_cap is not None:
                cache._evict_to_cap(prefill_hot_cap)

            torch.cuda.empty_cache()

        # Re-enable cold retrieval for generation
        cache._suppress_cold = False

        # Final eviction to hot_budget for generation phase
        if prefill_hot_cap is None:
            cache.mark_prefill_complete()
        else:
            cache._prefill_complete = True
            cache._evict_excess_all_layers()

        return logits

    # ── Attention function registration ──

    def _resolve_attn_impl(self) -> str:
        """Get the model's current attention implementation name."""
        config = self.model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        return getattr(config, "_attn_implementation", "eager")

    def _get_attn_fn(self, impl: str):
        """Get the attention function for a given implementation."""
        if impl in ALL_ATTENTION_FUNCTIONS:
            return ALL_ATTENTION_FUNCTIONS[impl]
        # For "eager", the model uses its own module-level function.
        # Look it up from the first global attention module.
        attn_module = self._text_model.layers[
            self.topology.global_layer_indices[0]
        ].self_attn
        model_module_name = type(attn_module).__module__
        import importlib
        model_module = importlib.import_module(model_module_name)
        eager_fn = getattr(model_module, "eager_attention_forward", None)
        if eager_fn is not None:
            return eager_fn
        raise ValueError(
            f"Cannot find attention function for implementation '{impl}'"
        )

    def _set_attn_impl(self, impl: str) -> None:
        """Set the attention implementation on the model config."""
        config = self.model.config
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = impl
        config._attn_implementation = impl

    def _make_kiv_attention(self, original_fn):
        """Build the KIV attention wrapper that augments with cold retrievals."""
        kiv_config = self.config

        # Compute fallback scaling from model (handles Gemma4's scaling=1.0)
        sample_attn = self._text_model.layers[
            self.topology.global_layer_indices[0]
        ].self_attn
        if hasattr(sample_attn, "scaling") and sample_attn.scaling is not None:
            fallback_scaling = sample_attn.scaling
        else:
            fallback_scaling = self.topology.head_dim ** -0.5

        def kiv_attention(
            module, query, key, value, attention_mask, **kwargs
        ):
            cache = getattr(module, "_kiv_cache", None)
            layer_idx = getattr(module, "_kiv_layer_idx", None)

            if cache is not None and layer_idx is not None:
                cold_store = cache.get_cold_store(layer_idx)

                if (
                    cold_store is not None
                    and cold_store.cold_length > 0
                    and not getattr(cache, "_suppress_cold", False)
                ):
                    scaling = kwargs.get("scaling")
                    if scaling is None:
                        scaling = fallback_scaling
                    cold_k, cold_v = cold_store.retrieve_top_kv(
                        query, scaling, kiv_config
                    )

                    if cold_k is not None:
                        # Crop mask to match hot K/V size (mask was
                        # created before eviction, so it may be wider)
                        hot_len = key.shape[2]
                        if (
                            attention_mask is not None
                            and attention_mask.shape[-1] > hot_len
                        ):
                            attention_mask = attention_mask[
                                ..., -hot_len:
                            ]

                        # Prepend cold K/V to hot K/V
                        key = torch.cat([cold_k, key], dim=2)
                        value = torch.cat([cold_v, value], dim=2)

                        # Extend mask for cold tokens (all attendable)
                        if attention_mask is not None:
                            attention_mask = _extend_mask_for_cold(
                                attention_mask, cold_k.shape[2]
                            )

            return original_fn(
                module, query, key, value, attention_mask, **kwargs
            )

        return kiv_attention
