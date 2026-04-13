"""Tests for vLLM topology detection."""

from types import SimpleNamespace

from kiv.vllm.topology import (
    detect_topology_from_vllm,
    _detect_global_layers,
    _detect_kv_sharing,
    _detect_head_geometry,
)


def _make_hf_config(**kwargs):
    defaults = {
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "hidden_size": 2048,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_vllm_config(hf_config):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config)
    )




def test_detect_global_layers_with_layer_types():
    config = _make_hf_config(
        layer_types=["sliding_attention", "full_attention"] * 13,
    )
    global_indices = _detect_global_layers(config)
    assert global_indices == tuple(range(1, 26, 2))


def test_detect_global_layers_all_global_no_sliding():
    config = _make_hf_config()
    global_indices = _detect_global_layers(config)
    assert global_indices == tuple(range(26))


def test_detect_global_layers_uniform_sliding_window():
    config = _make_hf_config(sliding_window=4096)
    global_indices = _detect_global_layers(config)
    assert global_indices == tuple(range(26))




def test_detect_kv_sharing_none():
    config = _make_hf_config()
    global_indices = tuple(range(26))
    independent, sharing_map = _detect_kv_sharing(config, global_indices)
    assert independent == global_indices
    assert sharing_map == {}


def test_detect_kv_sharing_gemma4_style():
    config = _make_hf_config(num_kv_shared_layers=12)
    global_indices = (0, 6, 12, 18, 24)
    independent, sharing_map = _detect_kv_sharing(config, global_indices)
    # shared_start = 26 - 12 = 14
    # layers < 14: 0, 6, 12 → independent
    # layers >= 14: 18, 24 → shared, source = 12 (last independent)
    assert independent == (0, 6, 12)
    assert sharing_map == {18: 12, 24: 12}




def test_detect_head_geometry_gqa():
    config = _make_hf_config(
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
    )
    q, kv, dim = _detect_head_geometry(config)
    assert q == 8
    assert kv == 2
    assert dim == 128


def test_detect_head_geometry_mha_fallback():
    config = _make_hf_config(
        num_attention_heads=32,
        hidden_size=4096,
    )
    # Remove num_key_value_heads to test MHA fallback
    del config.num_key_value_heads
    # Remove head_dim to test computed fallback
    del config.head_dim
    q, kv, dim = _detect_head_geometry(config)
    assert q == 32
    assert kv == 32  # MHA: same as query
    assert dim == 128  # 4096 / 32


def test_detect_head_geometry_global_specific():
    config = _make_hf_config(
        num_attention_heads=8,
        num_key_value_heads=4,
        num_global_key_value_heads=1,
        head_dim=128,
        global_head_dim=256,
    )
    q, kv, dim = _detect_head_geometry(config)
    assert kv == 1   # global-specific takes priority
    assert dim == 256  # global_head_dim takes priority




def test_detect_topology_from_vllm_basic():
    hf_config = _make_hf_config(
        model_type="gemma4_text",
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    vllm_config = _make_vllm_config(hf_config)
    topology = detect_topology_from_vllm(vllm_config)

    assert topology.model_family == "gemma4"
    assert topology.num_hidden_layers == 4
    assert topology.global_layer_indices == (1, 3)
    assert topology.num_query_heads == 8
    assert topology.num_kv_heads == 1
    assert topology.head_dim == 256


def test_detect_topology_from_vllm_unwraps_text_config():
    text_config = _make_hf_config(
        model_type="llama",
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
    )
    outer_config = SimpleNamespace(text_config=text_config)
    vllm_config = _make_vllm_config(outer_config)

    topology = detect_topology_from_vllm(vllm_config)
    assert topology.model_family == "llama"
    assert topology.num_hidden_layers == 2
    assert topology.num_query_heads == 16
