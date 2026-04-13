"""Attention hook for vLLM: two-pass cold K/V retrieval during decode."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..config import KIVConfig

if TYPE_CHECKING:
    from .connector import KIVConnector

logger = logging.getLogger(__name__)


def install_attention_hook(
    model: Any,
    connector: KIVConnector,
) -> dict[int, Any]:
    """Patch vLLM model attention layers to inject cold K/V retrieval.

    Wraps the inner ``Attention.forward`` within each global attention layer
    to add a second attention pass over cold-retrieved K/V, merged with
    the hot attention output.

    Args:
        model: The vLLM model instance.
        connector: KIVConnector instance owning the ColdKVStores.

    Returns:
        Dict mapping layer_idx to cleanup info for ``uninstall_attention_hook``.
    """
    originals: dict[int, dict[str, Any]] = {}
    topology = connector.topology

    layers = _find_layers(model)
    if layers is None:
        logger.warning(
            "Could not locate model layers for KIV attention hook. "
            "Cold retrieval will NOT be active during decode. "
            "Model type: %s",
            type(model).__name__,
        )
        return originals

    for layer_idx in topology.global_layer_indices:
        if layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        self_attn = _find_self_attn(layer)
        if self_attn is None:
            continue

        inner_attn = _find_inner_attn(self_attn)
        if inner_attn is None:
            logger.debug(
                "Layer %d: no inner Attention object found, skipping.",
                layer_idx,
            )
            continue

        # Guard against self_attn and inner_attn being the same object
        if inner_attn is self_attn:
            logger.debug(
                "Layer %d: inner_attn is self_attn (same object), skipping "
                "to avoid infinite recursion.",
                layer_idx,
            )
            continue

        # Save originals for cleanup
        original_inner_forward = inner_attn.forward

        # Patch the inner attention forward
        patched_inner = _make_kiv_inner_forward(
            original_forward=original_inner_forward,
            layer_idx=layer_idx,
            connector=connector,
        )
        inner_attn.forward = patched_inner

        originals[layer_idx] = {
            "inner_attn": inner_attn,
            "inner_forward": original_inner_forward,
        }
        logger.debug("KIV attention hook installed on layer %d", layer_idx)

    logger.info(
        "KIV attention hooks installed on %d/%d global layers.",
        len(originals),
        len(topology.global_layer_indices),
    )
    return originals


def uninstall_attention_hook(
    model: Any,
    originals: dict[int, dict[str, Any]],
) -> None:
    """Restore original attention forward methods."""
    for layer_idx, info in originals.items():
        inner_attn = info["inner_attn"]
        inner_attn.forward = info["inner_forward"]

    logger.info("KIV attention hooks uninstalled from %d layers.", len(originals))


def _find_layers(model: Any) -> list[Any] | None:
    """Locate the transformer layer list in a vLLM model."""
    for path in [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.text_model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = path(model)
            return list(layers)
        except AttributeError:
            continue
    return None


def _find_self_attn(layer: Any) -> Any | None:
    """Find the self-attention module within a transformer layer."""
    for attr in ("self_attn", "attn", "attention"):
        mod = getattr(layer, attr, None)
        if mod is not None:
            return mod
    return None


def _find_inner_attn(self_attn: Any) -> Any | None:
    for attr in ("attn", "attention", "_attn"):
        obj = getattr(self_attn, attr, None)
        if obj is not None and hasattr(obj, "forward"):
            cls_name = type(obj).__name__
            if "Attention" in cls_name or "attn" in cls_name.lower():
                return obj
    return None


def _make_kiv_inner_forward(
    original_forward: Any,
    layer_idx: int,
    connector: KIVConnector,
):
    """Wrap inner Attention.forward to run cold retrieval and merge."""
    kiv_config = connector.kiv_config
    head_dim = connector.topology.head_dim
    scaling = head_dim ** -0.5

    def kiv_inner_forward(query, key, value, *args, **kwargs):
        hot_output = original_forward(query, key, value, *args, **kwargs)

        cold_store = connector.get_cold_store(layer_idx)
        if (
            cold_store is None
            or cold_store.cold_length == 0
            or not connector._prefill_complete
        ):
            return hot_output

        q_for_retrieval = _reshape_query_for_retrieval(
            query, connector.topology.num_query_heads, head_dim
        )
        if q_for_retrieval is None:
            return hot_output

        cold_k, cold_v = cold_store.retrieve_top_kv(
            q_for_retrieval,
            scaling,
            kiv_config,
            step=connector._decode_step,
        )

        if cold_k is None:
            return hot_output

        cold_output = _cold_attention(
            q_for_retrieval, cold_k, cold_v, scaling,
            connector.topology.num_query_heads,
            connector.topology.num_kv_heads,
        )

        cold_output_flat = _reshape_output_to_match(
            cold_output, hot_output.shape
        )
        if cold_output_flat is None:
            return hot_output

        # equal weight merge (proper log-sum-exp merge needs attention backend changes)
        merged = 0.5 * hot_output + 0.5 * cold_output_flat
        return merged

    return kiv_inner_forward


def _reshape_query_for_retrieval(
    query: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor | None:
    """Reshape vLLM query tensor to ColdKVStore format [B, H, Q, D]."""
    if query.ndim == 3:
        # [num_tokens, num_heads, head_dim] -> [1, num_heads, num_tokens, head_dim]
        return query.permute(1, 0, 2).unsqueeze(0)
    elif query.ndim == 2:
        # [num_tokens, num_heads * head_dim] -> reshape
        num_tokens = query.shape[0]
        try:
            q = query.view(num_tokens, num_heads, head_dim)
            return q.permute(1, 0, 2).unsqueeze(0)
        except RuntimeError:
            logger.debug(
                "Cannot reshape query %s to [T, %d, %d]",
                query.shape, num_heads, head_dim,
            )
            return None
    elif query.ndim == 4:
        # Already [B, H, Q, D]
        return query
    else:
        logger.debug("Unexpected query shape: %s", query.shape)
        return None


def _cold_attention(
    query: torch.Tensor,
    cold_k: torch.Tensor,
    cold_v: torch.Tensor,
    scaling: float,
    num_query_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    num_kv_groups = num_query_heads // num_kv_heads
    if num_kv_groups > 1:
        B, H_kv, P, D = cold_k.shape
        cold_k = (
            cold_k[:, :, None, :, :]
            .expand(B, H_kv, num_kv_groups, P, D)
            .reshape(B, num_query_heads, P, D)
        )
        cold_v = (
            cold_v[:, :, None, :, :]
            .expand(B, H_kv, num_kv_groups, P, D)
            .reshape(B, num_query_heads, P, D)
        )

    # Scaled dot-product attention
    attn_weights = torch.matmul(query, cold_k.transpose(-2, -1)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, cold_v)

    return output


def _reshape_output_to_match(
    cold_output: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor | None:
    # cold_output [B, H, Q, D] -> target layout
    if cold_output.shape == target_shape:
        return cold_output

    try:
        if len(target_shape) == 2:
            # Target: [num_tokens, hidden_dim]
            B, H, Q, D = cold_output.shape
            return cold_output.squeeze(0).permute(1, 0, 2).reshape(Q, H * D)
        elif len(target_shape) == 3:
            # Target: [num_tokens, num_heads, head_dim]
            return cold_output.squeeze(0).permute(1, 0, 2)
        else:
            return cold_output.view(target_shape)
    except RuntimeError:
        logger.debug(
            "Cannot reshape cold output %s to %s",
            cold_output.shape, target_shape,
        )
        return None
