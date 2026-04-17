"""Tiered KV cache: extends DynamicCache with cold eviction for global layers."""

from __future__ import annotations

import logging

import torch
from transformers import DynamicCache

from .cold_store import ColdKVStore
from .config import KIVConfig
from .model_topology import ModelTopology

logger = logging.getLogger(__name__)


class TieredKVCache(DynamicCache):

    def __init__(
        self,
        config,
        kiv_config: KIVConfig,
        topology: ModelTopology,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config=config)
        self.kiv_config = kiv_config
        self.topology = topology

        if device is None:
            device = torch.device("cuda")

        # One cold store per independent KV layer
        self.cold_stores: dict[int, ColdKVStore] = {}
        for layer_idx in topology.independent_kv_layers:
            self.cold_stores[layer_idx] = ColdKVStore(kiv_config, topology, device)

        # Track whether we're past prefill (eviction only starts after)
        self._prefill_complete = False
        # When True, attention function skips cold retrieval (hot-only attention)
        self._suppress_cold = False
        # Track total tokens per layer (hot + cold) for position tracking
        self._total_tokens: dict[int, int] = {}
        # Decode step counter for cold store candidate caching
        self._decode_step: int = 0

    def _kv_source_layer_idx(self, layer_idx: int) -> int:
        """Resolve shared global layers to the layer that owns their KV cache."""
        return self.topology.kv_sharing_map.get(layer_idx, layer_idx)

    def mark_prefill_complete(self) -> None:
        self._prefill_complete = True
        self._evict_excess_all_layers()

    def _evict_layer(self, layer_idx: int, num_evict: int) -> None:
        """Evict the oldest num_evict tokens from a layer's hot cache to cold.

        Slices K/V, copies to cold store, crops the hot cache, and updates
        total token tracking. Used by all eviction paths.
        """
        layer = self.layers[layer_idx]
        k_evicted = layer.keys[:, :, :num_evict, :].contiguous()
        v_evicted = layer.values[:, :, :num_evict, :].contiguous()

        self.cold_stores[layer_idx].evict_from_hot(k_evicted, v_evicted)

        layer.keys = layer.keys[:, :, num_evict:, :].contiguous()
        layer.values = layer.values[:, :, num_evict:, :].contiguous()

        cold_len = self.cold_stores[layer_idx].cold_length
        hot_len = layer.keys.shape[2]
        self._total_tokens[layer_idx] = cold_len + hot_len

    def _evict_to_cap(self, cap: int) -> None:
        """Evict tokens exceeding cap from global layers. Used during bounded prefill."""
        for layer_idx in self.topology.independent_kv_layers:
            if layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            if not layer.is_initialized:
                continue

            seq_len = layer.keys.shape[2]
            if seq_len > cap:
                self._evict_layer(layer_idx, seq_len - cap)

    def _evict_excess_all_layers(self) -> None:
        """Evict tokens exceeding hot_budget from all independent global layers."""
        budget = self.kiv_config.hot_budget
        for layer_idx in self.topology.independent_kv_layers:
            if layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            if not layer.is_initialized:
                continue

            seq_len = layer.keys.shape[2]
            if seq_len > budget:
                num_evict = seq_len - budget
                self._evict_layer(layer_idx, num_evict)
                logger.debug(
                    "Layer %d: evicted %d tokens to cold (hot=%d, cold=%d)",
                    layer_idx, num_evict, budget,
                    self.cold_stores[layer_idx].cold_length,
                )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)

        if layer_idx in self.topology.independent_kv_layers:
            if (
                self._prefill_complete
                and self.topology.independent_kv_layers
                and layer_idx == self.topology.independent_kv_layers[0]
            ):
                self._decode_step += 1
            cold_len = self.cold_stores[layer_idx].cold_length
            hot_len = keys.shape[2]
            self._total_tokens[layer_idx] = cold_len + hot_len

        # Only evict for independent global layers, and only after prefill
        if (
            self._prefill_complete
            and layer_idx in self.cold_stores
        ):
            layer = self.layers[layer_idx]
            seq_len = layer.keys.shape[2]
            budget = self.kiv_config.hot_budget

            if seq_len > budget:
                self._evict_layer(layer_idx, seq_len - budget)
                keys = layer.keys
                values = layer.values

        return keys, values

    def get_cold_store(self, layer_idx: int) -> ColdKVStore | None:
        """
        Get cold store for a layer, resolving shared layers to their source.
        """
        if layer_idx in self.cold_stores:
            return self.cold_stores[layer_idx]
        # Check if this is a shared global layer
        source = self.topology.kv_sharing_map.get(layer_idx)
        if source is not None:
            return self.cold_stores.get(source)
        return None

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        cold_store = self.get_cold_store(layer_idx)
        source_layer_idx = self._kv_source_layer_idx(layer_idx)
        if cold_store is not None and source_layer_idx < len(self.layers):
            cold_len = cold_store.cold_length
            if cold_len > 0:
                layer = self.layers[source_layer_idx]
                hot_len = layer.get_seq_length() if layer.is_initialized else 0
                kv_length = hot_len + query_length
                kv_offset = cold_len
                return kv_length, kv_offset

        return super().get_mask_sizes(query_length, layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns hot + cold length so position_ids stay correct."""
        if layer_idx in self._total_tokens:
            return self._total_tokens[layer_idx]

        source_layer_idx = self._kv_source_layer_idx(layer_idx)
        if source_layer_idx in self._total_tokens:
            return self._total_tokens[source_layer_idx]
        if source_layer_idx != layer_idx:
            return super().get_seq_length(source_layer_idx)

        # For sliding/other layers, use parent behavior
        return super().get_seq_length(layer_idx)

    def reset(self) -> None:
        """Clear all state."""
        super().reset()
        for store in self.cold_stores.values():
            store.reset()
        self._total_tokens.clear()
        self._prefill_complete = False
        self._suppress_cold = False
        self._decode_step = 0

    def truncate_to(self, new_len: int) -> bool:
        """Trim each global layer to ``new_len`` tokens.

        Returns ``True`` on success, ``False`` if the requested length
        would require dropping tokens that have already been evicted
        to cold storage. In that case the cache is left untouched.

        The operation is intended for cases like chat regeneration,
        where the client sends a request that is a strict prefix of
        what the cache currently holds and the caller wants to reuse
        the cache up to ``new_len`` instead of re-prefilling from
        scratch. Keeping the truncation inside the hot window avoids
        the complexity of "un-evicting" pages from cold.

        Per-layer behaviour:

        * ``new_len > current_total`` - rejected (cannot extend).
        * ``new_len < cold_length`` - rejected (would require cold
          truncation, which is not implemented).
        * ``new_len`` within ``[cold_length, cold_length + hot_length]``
          - hot K/V tensors are sliced to ``new_len - cold_length``
          positions and total-token tracking is updated.

        The cold store's candidate cache and partial-page buffers are
        left intact: truncation only affects hot, and the partial
        buffer contains tokens that are already counted in
        ``cold_length``. The decode-step counter is reset so shared
        layer candidate caching starts fresh on the next forward.
        """
        for layer_idx in self.topology.independent_kv_layers:
            if layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            if not layer.is_initialized:
                continue
            cold_len = self.cold_stores[layer_idx].cold_length
            hot_len = layer.keys.shape[2]
            current_total = cold_len + hot_len
            if new_len > current_total:
                return False
            if new_len < cold_len:
                return False

        for layer_idx in self.topology.independent_kv_layers:
            if layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            if not layer.is_initialized:
                continue
            cold_len = self.cold_stores[layer_idx].cold_length
            target_hot_len = new_len - cold_len
            if target_hot_len < layer.keys.shape[2]:
                layer.keys = layer.keys[:, :, :target_hot_len, :].contiguous()
                layer.values = layer.values[:, :, :target_hot_len, :].contiguous()
            self._total_tokens[layer_idx] = new_len

        for store in self.cold_stores.values():
            store._cached_candidate_k = None
            store._cached_candidate_v = None
            store._cached_candidate_idx = None
            store._cached_step = -1

        self._decode_step = 0
        return True

    def memory_report(self) -> dict:
        """Report memory usage across all tiers."""
        hot_bytes = 0
        for layer_idx in self.topology.independent_kv_layers:
            if layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
                layer = self.layers[layer_idx]
                hot_bytes += layer.keys.nelement() * layer.keys.element_size()
                hot_bytes += layer.values.nelement() * layer.values.element_size()

        k_index_cpu = 0
        v_store_cpu = 0
        page_summaries_gpu = 0
        partial_gpu = 0
        for store in self.cold_stores.values():
            mem = store.memory_bytes()
            k_index_cpu += mem["k_index_cpu"]
            v_store_cpu += mem["v_store_cpu"]
            page_summaries_gpu += mem["page_summaries_gpu"]
            partial_gpu += mem["partial_gpu"]

        return {
            "hot_vram_bytes": hot_bytes,
            "k_index_cpu_bytes": k_index_cpu,
            "v_store_cpu_bytes": v_store_cpu,
            "page_summaries_gpu_bytes": page_summaries_gpu,
            "total_vram_bytes": hot_bytes + page_summaries_gpu + partial_gpu,
            "total_cpu_bytes": k_index_cpu + v_store_cpu,
        }
