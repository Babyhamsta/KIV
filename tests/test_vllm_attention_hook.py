"""Tests for vLLM attention hook helpers."""

import torch
import pytest

from kiv.vllm.attention_hook import (
    _reshape_query_for_retrieval,
    _cold_attention,
    _reshape_output_to_match,
    _find_layers,
    _find_self_attn,
    _find_inner_attn,
)




def test_reshape_query_3d():
    """[num_tokens, num_heads, head_dim] → [1, num_heads, num_tokens, head_dim]"""
    q = torch.randn(5, 8, 64)
    result = _reshape_query_for_retrieval(q, num_heads=8, head_dim=64)
    assert result is not None
    assert result.shape == (1, 8, 5, 64)


def test_reshape_query_2d():
    """[num_tokens, num_heads * head_dim] → [1, num_heads, num_tokens, head_dim]"""
    q = torch.randn(3, 512)  # 8 heads * 64 dim
    result = _reshape_query_for_retrieval(q, num_heads=8, head_dim=64)
    assert result is not None
    assert result.shape == (1, 8, 3, 64)


def test_reshape_query_4d_passthrough():
    """Already [B, H, Q, D] — return as-is."""
    q = torch.randn(1, 8, 1, 64)
    result = _reshape_query_for_retrieval(q, num_heads=8, head_dim=64)
    assert result is not None
    assert result.shape == (1, 8, 1, 64)


def test_reshape_query_bad_shape_returns_none():
    q = torch.randn(2, 3, 4, 5, 6)  # 5D — unsupported
    result = _reshape_query_for_retrieval(q, num_heads=8, head_dim=64)
    assert result is None


def test_reshape_query_2d_incompatible_returns_none():
    """Dimensions don't divide evenly into num_heads * head_dim."""
    q = torch.randn(3, 100)  # 100 != 8*64
    result = _reshape_query_for_retrieval(q, num_heads=8, head_dim=64)
    assert result is None




def test_cold_attention_basic():
    """GQA: 8 query heads, 2 KV heads."""
    B, H_q, Q, D = 1, 8, 1, 64
    H_kv, P = 2, 16
    query = torch.randn(B, H_q, Q, D)
    cold_k = torch.randn(B, H_kv, P, D)
    cold_v = torch.randn(B, H_kv, P, D)

    output = _cold_attention(query, cold_k, cold_v, 0.125, H_q, H_kv)
    assert output.shape == (B, H_q, Q, D)


def test_cold_attention_mha():
    """MHA: 4 heads, no GQA expansion."""
    B, H, Q, D = 1, 4, 1, 32
    P = 10
    query = torch.randn(B, H, Q, D)
    cold_k = torch.randn(B, H, P, D)
    cold_v = torch.randn(B, H, P, D)

    output = _cold_attention(query, cold_k, cold_v, 0.125, H, H)
    assert output.shape == (B, H, Q, D)


def test_cold_attention_mqa():
    """MQA: 1 KV head, many query heads."""
    B, H_q, Q, D = 1, 8, 1, 64
    H_kv, P = 1, 20
    query = torch.randn(B, H_q, Q, D)
    cold_k = torch.randn(B, H_kv, P, D)
    cold_v = torch.randn(B, H_kv, P, D)

    output = _cold_attention(query, cold_k, cold_v, 0.125, H_q, H_kv)
    assert output.shape == (B, H_q, Q, D)




def test_reshape_output_to_2d():
    """[1, 8, 1, 64] → [1, 512]"""
    cold_out = torch.randn(1, 8, 1, 64)
    target = torch.Size([1, 512])
    result = _reshape_output_to_match(cold_out, target)
    assert result is not None
    assert result.shape == target


def test_reshape_output_to_3d():
    """[1, 8, 1, 64] → [1, 8, 64]"""
    cold_out = torch.randn(1, 8, 1, 64)
    target = torch.Size([1, 8, 64])
    result = _reshape_output_to_match(cold_out, target)
    assert result is not None
    assert result.shape == target


def test_reshape_output_same_shape():
    cold_out = torch.randn(1, 8, 1, 64)
    result = _reshape_output_to_match(cold_out, cold_out.shape)
    assert result is not None
    assert result.shape == cold_out.shape


def test_reshape_output_incompatible_returns_none():
    cold_out = torch.randn(1, 8, 1, 64)
    # 5D target with incompatible element count — view fails
    result = _reshape_output_to_match(cold_out, torch.Size([1, 2, 3, 5, 7]))
    assert result is None




def test_find_layers_model_dot_model():
    layers = [object(), object()]
    model = type("Model", (), {"model": type("Inner", (), {"layers": layers})()})()
    result = _find_layers(model)
    assert result is not None
    assert len(result) == 2


def test_find_layers_direct():
    layers = [object()]
    model = type("Model", (), {"layers": layers})()
    result = _find_layers(model)
    assert result is not None
    assert len(result) == 1


def test_find_layers_none():
    model = type("Model", (), {})()
    result = _find_layers(model)
    assert result is None


def test_find_self_attn():
    attn = object()
    layer = type("Layer", (), {"self_attn": attn})()
    assert _find_self_attn(layer) is attn


def test_find_self_attn_alt_name():
    attn = object()
    layer = type("Layer", (), {"attn": attn})()
    assert _find_self_attn(layer) is attn


def test_find_self_attn_none():
    layer = type("Layer", (), {})()
    assert _find_self_attn(layer) is None


def test_find_inner_attn():
    inner = type("Attention", (), {"forward": lambda: None})()
    self_attn = type("SelfAttn", (), {"attn": inner})()
    assert _find_inner_attn(self_attn) is inner


def test_find_inner_attn_same_object_returns_none():
    """Circular ref: attn.attn = attn. Finder returns it, caller checks identity."""
    # _find_inner_attn itself doesn't check identity — install_attention_hook does
    attn = type("Attention", (), {"forward": lambda: None, "attn": None})()
    attn.attn = attn  # circular
    result = _find_inner_attn(attn)
    assert result is attn  # finds it — caller checks identity
