"""
KIV Middleware: installs tiered KV cache into Gemma 4 E2B without
modifying model weights. Monkey-patches attention forward for global layers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from .config import KIVConfig
from .partitioned_attn import two_partition_attention
from .tiered_cache import TieredKVCache

logger = logging.getLogger(__name__)


class KIVMiddleware:
    """
    Installs K-Indexed V Materialization into a Gemma 4 E2B model.

    Does not modify model weights. Patches attention forward functions
    for the 7 global attention layers to use two-partition attention
    with the TieredKVCache.

    Usage:
        middleware = KIVMiddleware(model, config)
        middleware.install()
        cache = middleware.create_cache()
        output = model.generate(..., past_key_values=cache)
        middleware.uninstall()
    """

    def __init__(self, model: Any, config: KIVConfig | None = None) -> None:
        self.model = model
        self.config = config or KIVConfig()
        self._text_model = self._find_text_model()
        self._original_forwards: dict[int, Callable] = {}
        self._installed = False

    def _find_text_model(self):
        """Navigate HF model hierarchy to find text model with layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model
        raise AttributeError("Cannot find text model layers in model hierarchy.")

    def install(self) -> None:
        """Patch attention forward for all global layers."""
        if self._installed:
            logger.warning("KIV middleware already installed.")
            return

        for layer_idx in self.config.global_layer_indices:
            attn = self._text_model.layers[layer_idx].self_attn
            self._original_forwards[layer_idx] = attn.forward
            attn.forward = self._make_kiv_forward(attn, layer_idx)
            logger.debug("Patched layer %d attention with KIV forward", layer_idx)

        self._installed = True
        logger.info(
            "KIV middleware installed on %d global layers (hot=%d, top_p=%d)",
            len(self.config.global_layer_indices),
            self.config.hot_budget,
            self.config.top_p,
        )

    def uninstall(self) -> None:
        """Restore original attention forward methods."""
        for layer_idx, orig in self._original_forwards.items():
            self._text_model.layers[layer_idx].self_attn.forward = orig
        self._original_forwards.clear()
        self._installed = False
        logger.info("KIV middleware uninstalled.")

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
                attended during prefill (evict-only, no two-partition).
                This bounds VRAM at the cost of losing exact attention to
                early tokens. If None, hot cache grows unbounded.

        Returns:
            logits for the last token: [B, vocab_size]
        """
        B, T = input_ids.shape
        num_chunks = (T + chunk_size - 1) // chunk_size
        logits = None

        # During bounded prefill, suppress two-partition attention.
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

        # Re-enable cold attention for generation
        cache._suppress_cold = False

        # Final eviction to hot_budget for generation phase
        if prefill_hot_cap is None:
            cache.mark_prefill_complete()
        else:
            cache._prefill_complete = True
            cache._evict_excess_all_layers()

        return logits

    def create_cache(self, device: torch.device | None = None) -> TieredKVCache:
        """Create a TieredKVCache configured for this model."""
        if device is None:
            device = next(self.model.parameters()).device
        return TieredKVCache(
            config=self.model.config,
            kiv_config=self.config,
            device=device,
        )

    def _make_kiv_forward(
        self, attn_module: Any, layer_idx: int
    ) -> Callable:
        """
        Create a replacement forward for one global attention layer.

        The replacement:
        1. Computes Q, K, V using the original projections
        2. For non-shared: calls cache.update() (with eviction)
        3. For shared: reads hot K,V from shared_kv_states
        4. If cold entries exist: two_partition_attention()
        5. Else: standard eager attention
        6. Output projection and return
        """
        config = self.config
        orig_forward = self._original_forwards[layer_idx]

        # Import here to avoid circular imports at module level
        import transformers.models.gemma4.modeling_gemma4 as gemma4_mod
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        apply_rotary_pos_emb = gemma4_mod.apply_rotary_pos_emb
        eager_attention_forward = gemma4_mod.eager_attention_forward

        # Resolve attention implementation (sdpa, flash_attention_2, or eager)
        attn_impl = attn_module.config._attn_implementation
        if attn_impl != "eager":
            default_attention_forward = ALL_ATTENTION_FUNCTIONS[attn_impl]
        else:
            default_attention_forward = eager_attention_forward

        # Capture attention module attributes
        is_shared = attn_module.is_kv_shared_layer
        kv_shared_idx = attn_module.kv_shared_layer_index
        store_full_kv = attn_module.store_full_length_kv
        head_dim = attn_module.head_dim
        scaling = attn_module.scaling  # 1.0 for Gemma 4 (QK norm handles magnitude)

        def kiv_forward(
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: torch.Tensor | None,
            shared_kv_states: dict,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)

            cos, sin = position_embeddings

            # ── Q projection (always computed) ──
            query_states = attn_module.q_proj(hidden_states).view(hidden_shape)
            query_states = attn_module.q_norm(query_states)
            query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
            query_states = query_states.transpose(1, 2)  # [B, H_q, Q, D]

            # ── K, V (shared vs independent) ──
            if is_shared:
                key_states, value_states = shared_kv_states[kv_shared_idx]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                key_states = attn_module.k_proj(hidden_states).view(hidden_shape)
                v_proj = getattr(attn_module, "v_proj", None)
                if v_proj is not None:
                    value_states = v_proj(hidden_states).view(hidden_shape)
                else:
                    # attention_k_eq_v: V = k_proj output (before norms)
                    value_states = key_states

                key_states = attn_module.k_norm(key_states)
                key_states = apply_rotary_pos_emb(
                    key_states, cos, sin, unsqueeze_dim=2
                )
                key_states = key_states.transpose(1, 2)  # [B, H_kv, T, D]

                value_states = attn_module.v_norm(value_states)
                value_states = value_states.transpose(1, 2)  # [B, H_kv, T, D]

                # Cache update (handles eviction for TieredKVCache)
                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, layer_idx
                    )

            # Store hot KV for shared layers (layer 14 only)
            if store_full_kv:
                shared_kv_states[layer_idx] = key_states, value_states

            # ── Decide: standard attention or two-partition ──
            cold_store = None
            if past_key_values is not None and isinstance(past_key_values, TieredKVCache):
                cold_store = past_key_values.get_cold_store(layer_idx)

            use_partitioned = (
                cold_store is not None
                and cold_store.cold_length > 0
                and not getattr(past_key_values, "_suppress_cold", False)
            )

            if use_partitioned:
                # Crop mask to match hot cache size (mask was created
                # before eviction, so it may be larger)
                hot_len = key_states.shape[2]
                hot_mask = attention_mask
                if hot_mask is not None and hot_mask.shape[-1] > hot_len:
                    # Keep only the last hot_len columns (most recent tokens)
                    hot_mask = hot_mask[:, :, :, -hot_len:]

                # Two-partition attention
                attn_output = two_partition_attention(
                    query=query_states,
                    hot_key=key_states,
                    hot_value=value_states,
                    cold_store=cold_store,
                    hot_mask=hot_mask,
                    config=config,
                    scaling=scaling,
                )
                # attn_output: [B, H_q, Q, D]
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_weights = None
            else:
                # Standard attention (sdpa/flash/eager based on model config)
                attn_output, attn_weights = default_attention_forward(
                    attn_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0,
                    scaling=scaling,
                    sliding_window=None,  # global layers have no window
                    **kwargs,
                )

            # ── Output projection ──
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            return attn_output, attn_weights

        return kiv_forward
