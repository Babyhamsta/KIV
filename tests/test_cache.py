"""Regression tests for KIV cache behavior."""

import torch
from transformers import PretrainedConfig

from kiv.cold_store import ColdKVStore
from kiv.config import KIVConfig
from kiv.model_topology import ModelTopology
from kiv.tiered_cache import TieredKVCache


def _topology(
    *,
    global_layer_indices=(0,),
    independent_kv_layers=(0,),
    kv_sharing_map=None,
    num_hidden_layers=1,
):
    return ModelTopology.manual(
        global_layer_indices=global_layer_indices,
        independent_kv_layers=independent_kv_layers,
        kv_sharing_map=kv_sharing_map or {},
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        num_hidden_layers=num_hidden_layers,
    )


def test_retrieve_top_kv_preserves_batch_dimension():
    config = KIVConfig(page_size=2, top_p=1, top_pages=1, prefetch_stream=False)
    store = ColdKVStore(config, _topology(), torch.device("cpu"))

    key = torch.randn(2, 1, 2, 4)
    value = torch.randn(2, 1, 2, 4)
    query = torch.randn(2, 2, 1, 4)

    store.evict_from_hot(key, value)
    cold_key, cold_value = store.retrieve_top_kv(query, 1.0, config)

    assert cold_key.shape == (2, 1, 1, 4)
    assert cold_value.shape == (2, 1, 1, 4)
    hot_key = torch.randn(2, 1, 1, 4)
    assert torch.cat([cold_key, hot_key], dim=2).shape == (2, 1, 2, 4)


def test_shared_layer_seq_length_resolves_to_source_layer():
    topology = _topology(
        global_layer_indices=(0, 1),
        independent_kv_layers=(0,),
        kv_sharing_map={1: 0},
        num_hidden_layers=2,
    )
    cache = TieredKVCache(
        PretrainedConfig(num_hidden_layers=2),
        KIVConfig(prefetch_stream=False),
        topology,
        torch.device("cpu"),
    )

    cache._total_tokens[0] = 123

    assert cache.get_seq_length(0) == 123
    assert cache.get_seq_length(1) == 123


def test_truncate_to_hot_only_succeeds():
    """Truncation within the hot window slices K/V tensors in place."""
    topology = _topology()
    cache = TieredKVCache(
        PretrainedConfig(num_hidden_layers=1),
        KIVConfig(prefetch_stream=False),
        topology,
        torch.device("cpu"),
    )

    key = torch.randn(1, 1, 10, 4)
    value = torch.randn(1, 1, 10, 4)
    cache.update(key, value, 0)

    assert cache.truncate_to(6) is True

    layer = cache.layers[0]
    assert layer.keys.shape[2] == 6
    assert layer.values.shape[2] == 6
    assert cache._total_tokens[0] == 6
    # Original content at the kept positions is preserved.
    torch.testing.assert_close(layer.keys, key[:, :, :6, :])
    torch.testing.assert_close(layer.values, value[:, :, :6, :])


def test_truncate_to_rejects_lengths_beyond_current_total():
    topology = _topology()
    cache = TieredKVCache(
        PretrainedConfig(num_hidden_layers=1),
        KIVConfig(prefetch_stream=False),
        topology,
        torch.device("cpu"),
    )

    key = torch.randn(1, 1, 4, 4)
    value = torch.randn(1, 1, 4, 4)
    cache.update(key, value, 0)

    assert cache.truncate_to(10) is False
    # Cache state is unchanged on rejection.
    assert cache.layers[0].keys.shape[2] == 4


def test_truncate_to_rejects_truncation_into_cold():
    """Rolling back past the cold boundary is not supported."""
    config = KIVConfig(hot_budget=4, page_size=2, prefetch_stream=False)
    topology = _topology()
    cache = TieredKVCache(
        PretrainedConfig(num_hidden_layers=1),
        config,
        topology,
        torch.device("cpu"),
    )

    key = torch.randn(1, 1, 10, 4)
    value = torch.randn(1, 1, 10, 4)
    cache.update(key, value, 0)
    cache.mark_prefill_complete()

    # With hot_budget=4, 6 tokens now live in cold storage. Requesting
    # truncation below cold_length must be rejected.
    assert cache.cold_stores[0].cold_length == 6
    assert cache.truncate_to(3) is False
    # Truncation down to exactly cold_length is allowed (hot empties).
    assert cache.truncate_to(6) is True
    assert cache.layers[0].keys.shape[2] == 0
