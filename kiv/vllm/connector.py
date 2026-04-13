"""KIV KV Connector for vLLM V1."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

from ..cold_store import ColdKVStore
from ..config import KIVConfig
from ..model_topology import ModelTopology
from .topology import detect_topology_from_vllm

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVCacheConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )

logger = logging.getLogger(__name__)


@dataclass
class KIVConnectorMetadata:
    """Scheduler-to-worker metadata (minimal for KIV)."""
    pass


def _parse_layer_index(layer_name: str) -> int | None:
    """Extract numeric layer index from vLLM layer names.

    vLLM uses names like 'model.layers.0.self_attn', 'model.layers.12.self_attn',
    etc. We extract the layer number.
    """
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    return None


def _get_base_class() -> type:
    """Import KVConnectorBase_V1 at runtime so kiv core works without vllm."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        return KVConnectorBase_V1
    except ImportError:
        return object


class KIVConnector(_get_base_class()):
    """Shadows KV data into ColdKVStore on CPU for cold retrieval during decode."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        self._vllm_config = vllm_config
        self._role = role
        self._kv_cache_config = kv_cache_config

        # Read KIV config from vLLM's extra_config or use defaults
        extra = getattr(vllm_config, "extra_config", None) or {}
        kiv_extra = extra.get("kiv", {}) if isinstance(extra, dict) else {}
        self.kiv_config = KIVConfig(
            hot_budget=kiv_extra.get("hot_budget", 2048),
            top_p=kiv_extra.get("top_p", 256),
            page_size=kiv_extra.get("page_size", 128),
            top_pages=kiv_extra.get("top_pages", 32),
        )

        # Detect topology
        topology_cfg = kiv_extra.get("topology", None)
        if topology_cfg is not None:
            self.topology = ModelTopology.manual(**topology_cfg)
        else:
            self.topology = detect_topology_from_vllm(vllm_config)

        # Cold stores — one per independent KV layer
        self.cold_stores: dict[int, ColdKVStore] = {}
        self._device: torch.device | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}

        # Decode tracking
        self._prefill_complete = False
        self._decode_step = 0

        logger.info(
            "KIVConnector initialized: top_p=%d, page_size=%d, "
            "top_pages=%d, %d independent layers",
            self.kiv_config.top_p,
            self.kiv_config.page_size,
            self.kiv_config.top_pages,
            len(self.topology.independent_kv_layers),
        )

    def get_cold_store(self, layer_idx: int) -> ColdKVStore | None:
        if layer_idx in self.cold_stores:
            return self.cold_stores[layer_idx]
        source = self.topology.kv_sharing_map.get(layer_idx)
        if source is not None:
            return self.cold_stores.get(source)
        return None

    def _ensure_cold_store(self, layer_idx: int) -> ColdKVStore | None:
        if layer_idx not in self.topology.independent_kv_layers:
            return None
        if layer_idx not in self.cold_stores:
            if self._device is None:
                self._device = torch.device("cuda")
            self.cold_stores[layer_idx] = ColdKVStore(
                kiv_config=self.kiv_config,
                topology=self.topology,
                device=self._device,
            )
        return self.cold_stores[layer_idx]

    def start_load_kv(
        self, forward_context: Any, **kwargs: Any
    ) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Called after each layer's attention with the KV tensor.

        Shadows new KV data into the cold store. vLLM's hot cache is
        untouched — we only copy data for later retrieval.

        Args:
            layer_name: e.g. 'model.layers.12.self_attn'
            kv_layer: KV cache tensor for this layer
            attn_metadata: Attention metadata with slot/token info
        """
        layer_idx = _parse_layer_index(layer_name)
        if layer_idx is None:
            return
        if layer_idx not in self.topology.global_layer_indices:
            return

        cold_store = self._ensure_cold_store(layer_idx)
        if cold_store is None:
            return

        # Detect prefill → decode transition
        num_decode = getattr(attn_metadata, "num_decode_tokens", 0)
        if not self._prefill_complete and num_decode > 0:
            self._prefill_complete = True

        # Shadow new KV tokens to cold store
        self._shadow_kv_to_cold(layer_idx, kv_layer, attn_metadata, cold_store)

    def _shadow_kv_to_cold(
        self,
        layer_idx: int,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        cold_store: ColdKVStore,
    ) -> None:
        """Copy new KV data from this step into the cold store.

        vLLM's kv_layer tensor format varies by backend. This method
        handles common layouts and reshapes to ColdKVStore's expected
        [B, H_kv, T, D] format.

        Only new tokens (from this step's prefill or decode) are copied,
        not the entire cache.
        """
        num_prefill = getattr(attn_metadata, "num_prefill_tokens", 0)
        num_decode = getattr(attn_metadata, "num_decode_tokens", 0)
        new_tokens = num_prefill + num_decode

        if new_tokens == 0:
            return

        # Extract K and V for the new tokens
        # Common vLLM format: [2, total_slots, num_heads, head_dim]
        # where dim 0 is [K, V], and new tokens occupy the last new_tokens slots
        if kv_layer.ndim == 4 and kv_layer.shape[0] == 2:
            total_slots = kv_layer.shape[1]
            # New tokens are at the end of the slot range
            start = max(0, total_slots - new_tokens)
            k_new = kv_layer[0, start:total_slots]  # [new_tokens, H, D]
            v_new = kv_layer[1, start:total_slots]  # [new_tokens, H, D]

            # Reshape to [B, H, T, D] for ColdKVStore
            k_new = k_new.permute(1, 0, 2).unsqueeze(0)  # [1, H, T, D]
            v_new = v_new.permute(1, 0, 2).unsqueeze(0)  # [1, H, T, D]

            cold_store.evict_from_hot(k_new, v_new)

        elif kv_layer.ndim == 5 and kv_layer.shape[0] == 2:
            # [2, num_blocks, block_size, num_heads, head_dim]
            # Flatten blocks to tokens, then take the last new_tokens
            K_blocks = kv_layer[0]  # [num_blocks, block_size, H, D]
            V_blocks = kv_layer[1]
            num_blocks, block_size, H, D = K_blocks.shape
            k_flat = K_blocks.reshape(-1, H, D)  # [total_tokens, H, D]
            v_flat = V_blocks.reshape(-1, H, D)

            total = k_flat.shape[0]
            start = max(0, total - new_tokens)
            k_new = k_flat[start:total].permute(1, 0, 2).unsqueeze(0)
            v_new = v_flat[start:total].permute(1, 0, 2).unsqueeze(0)

            cold_store.evict_from_hot(k_new, v_new)

        else:
            logger.warning(
                "Unexpected kv_layer shape %s for layer %d. "
                "Skipping shadow copy. This may indicate a vLLM version "
                "incompatibility — check kiv.vllm docs for supported formats.",
                kv_layer.shape,
                layer_idx,
            )

    def wait_for_save(self) -> None:
        """Called after all layers have been saved. Update decode step."""
        if self._prefill_complete:
            self._decode_step += 1

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Called during initialization with the GPU KV cache buffers."""
        self._kv_caches = kv_caches
        if kv_caches:
            first_tensor = next(iter(kv_caches.values()))
            self._device = first_tensor.device

    def shutdown(self) -> None:
        self.cold_stores.clear()
        logger.info("KIVConnector shut down.")

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        """No-op for KIV."""
        pass

    def build_connector_meta(
        self, scheduler_output: Any
    ) -> KIVConnectorMetadata:
        """Return minimal metadata."""
        return KIVConnectorMetadata()

    # All optional methods (bind_connector_metadata, handle_preemptions,
    # get_finished, request_finished, etc.) are inherited from
    # KVConnectorBase_V1 with default no-op implementations.

    @property
    def role(self) -> KVConnectorRole:
        return self._role
