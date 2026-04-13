"""Tests for KIVConnector."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import pytest




def _install_vllm_mock():
    base_module = SimpleNamespace(
        KVConnectorBase_V1=object,
        KVConnectorRole=SimpleNamespace(SCHEDULER=0, WORKER=1),
    )
    connector_v1 = SimpleNamespace(base=base_module)
    connector_module = SimpleNamespace(v1=connector_v1)
    kv_transfer = SimpleNamespace(kv_connector=connector_module)
    distributed = SimpleNamespace(kv_transfer=kv_transfer)
    config_module = SimpleNamespace(
        VllmConfig=type("VllmConfig", (), {}),
        KVCacheConfig=type("KVCacheConfig", (), {}),
    )
    vllm_mock = SimpleNamespace(distributed=distributed, config=config_module)

    sys.modules["vllm"] = vllm_mock
    sys.modules["vllm.config"] = config_module
    sys.modules["vllm.distributed"] = distributed
    sys.modules["vllm.distributed.kv_transfer"] = kv_transfer
    sys.modules["vllm.distributed.kv_transfer.kv_connector"] = connector_module
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1"] = connector_v1
    sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"] = base_module


_install_vllm_mock()

from kiv.vllm.connector import KIVConnector, _parse_layer_index




def _make_vllm_config(**kiv_overrides):
    hf_config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        hidden_size=512,
    )
    extra = {"kiv": kiv_overrides} if kiv_overrides else {}
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        extra_config=extra,
    )


def _make_connector(**kiv_overrides):
    vllm_config = _make_vllm_config(**kiv_overrides)
    return KIVConnector(vllm_config, role=1)




def test_parse_layer_index_standard():
    assert _parse_layer_index("model.layers.12.self_attn") == 12


def test_parse_layer_index_zero():
    assert _parse_layer_index("model.layers.0.self_attn") == 0


def test_parse_layer_index_no_match():
    assert _parse_layer_index("embedding.weight") is None




def test_connector_default_config():
    c = _make_connector()
    assert c.kiv_config.hot_budget == 2048
    assert c.kiv_config.top_p == 256
    assert c.kiv_config.page_size == 128
    assert c.kiv_config.top_pages == 32


def test_connector_custom_config():
    c = _make_connector(hot_budget=1024, top_p=512)
    assert c.kiv_config.hot_budget == 1024
    assert c.kiv_config.top_p == 512


def test_connector_topology_detected():
    c = _make_connector()
    assert c.topology.model_family == "llama"
    assert c.topology.num_hidden_layers == 4
    assert c.topology.global_layer_indices == (0, 1, 2, 3)
    assert c.topology.num_query_heads == 8
    assert c.topology.num_kv_heads == 2




def test_ensure_cold_store_creates_for_independent_layer():
    c = _make_connector()
    store = c._ensure_cold_store(0)
    assert store is not None
    assert 0 in c.cold_stores


def test_ensure_cold_store_returns_none_for_non_global():
    c = _make_connector()
    # Manually set topology to have only layer 0 as independent
    from kiv.model_topology import ModelTopology
    c.topology = ModelTopology(
        model_family="test",
        num_hidden_layers=4,
        global_layer_indices=(0,),
        independent_kv_layers=(0,),
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )
    store = c._ensure_cold_store(1)
    assert store is None


def test_get_cold_store_resolves_sharing():
    c = _make_connector()
    from kiv.model_topology import ModelTopology
    c.topology = ModelTopology(
        model_family="test",
        num_hidden_layers=4,
        global_layer_indices=(0, 1, 2, 3),
        independent_kv_layers=(0, 1),
        kv_sharing_map={2: 0, 3: 1},
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )
    # Create store for layer 0
    c._ensure_cold_store(0)
    # Layer 2 shares with layer 0
    shared_store = c.get_cold_store(2)
    assert shared_store is c.cold_stores[0]




def test_save_kv_layer_ignores_non_global():
    c = _make_connector()
    from kiv.model_topology import ModelTopology
    c.topology = ModelTopology(
        model_family="test",
        num_hidden_layers=4,
        global_layer_indices=(0,),
        independent_kv_layers=(0,),
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )
    # Layer 1 is not global — should be ignored
    attn_meta = SimpleNamespace(num_decode_tokens=1, num_prefill_tokens=0)
    kv = torch.randn(2, 10, 2, 64)
    c.save_kv_layer("model.layers.1.self_attn", kv, attn_meta)
    assert 1 not in c.cold_stores


def test_save_kv_layer_shadows_prefill_tokens():
    c = _make_connector()
    attn_meta = SimpleNamespace(num_decode_tokens=0, num_prefill_tokens=5)
    # Shape: [2, 5, num_kv_heads=2, head_dim=64]
    kv = torch.randn(2, 5, 2, 64)
    c.save_kv_layer("model.layers.0.self_attn", kv, attn_meta)
    store = c.cold_stores.get(0)
    assert store is not None
    assert store.cold_length == 5


def test_save_kv_layer_detects_prefill_to_decode_transition():
    c = _make_connector()
    assert c._prefill_complete is False
    # Prefill step
    attn_meta_prefill = SimpleNamespace(num_decode_tokens=0, num_prefill_tokens=10)
    c.save_kv_layer("model.layers.0.self_attn", torch.randn(2, 10, 2, 64), attn_meta_prefill)
    assert c._prefill_complete is False
    # Decode step
    attn_meta_decode = SimpleNamespace(num_decode_tokens=1, num_prefill_tokens=0)
    c.save_kv_layer("model.layers.0.self_attn", torch.randn(2, 1, 2, 64), attn_meta_decode)
    assert c._prefill_complete is True




def test_get_num_new_matched_tokens_returns_zero():
    c = _make_connector()
    count, matched = c.get_num_new_matched_tokens(MagicMock(), 100)
    assert count == 0
    assert matched is False




def test_wait_for_save_increments_decode_step():
    c = _make_connector()
    c._prefill_complete = True
    assert c._decode_step == 0
    c.wait_for_save()
    assert c._decode_step == 1
    c.wait_for_save()
    assert c._decode_step == 2


def test_shutdown_clears_stores():
    c = _make_connector()
    c._ensure_cold_store(0)
    assert len(c.cold_stores) > 0
    c.shutdown()
    assert len(c.cold_stores) == 0
