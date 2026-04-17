"""Invariant tests for :class:`kiv.model_topology.ModelTopology`."""

from __future__ import annotations

import pytest

from kiv.model_topology import ModelTopology


def _make(**overrides):
    defaults = dict(
        model_family="test",
        num_hidden_layers=4,
        global_layer_indices=(0, 1, 2),
        independent_kv_layers=(0, 1, 2),
        kv_sharing_map={},
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    defaults.update(overrides)
    return ModelTopology(**defaults)


def test_valid_topology_constructs() -> None:
    topology = _make(
        global_layer_indices=(0, 1, 2, 3),
        independent_kv_layers=(0, 2),
        kv_sharing_map={1: 0, 3: 2},
    )
    assert topology.kv_sharing_map == {1: 0, 3: 2}


def test_independent_kv_layers_not_subset_rejected() -> None:
    with pytest.raises(ValueError, match="independent_kv_layers"):
        _make(
            global_layer_indices=(0, 1),
            independent_kv_layers=(0, 1, 5),
        )


def test_sharing_key_outside_globals_rejected() -> None:
    """A shared global layer must also be declared global so its
    attention module is bound to a KV cache."""
    with pytest.raises(ValueError, match="kv_sharing_map keys"):
        _make(
            global_layer_indices=(0, 2),
            independent_kv_layers=(0, 2),
            kv_sharing_map={1: 0},
        )


def test_sharing_source_outside_independents_rejected() -> None:
    """A share's source layer must own a cold store
    (i.e. be listed in independent_kv_layers)."""
    with pytest.raises(ValueError, match="kv_sharing_map values"):
        _make(
            global_layer_indices=(0, 1, 2),
            independent_kv_layers=(0,),
            kv_sharing_map={2: 1},
        )


def test_manual_defaults_to_globals_for_independents() -> None:
    topology = ModelTopology.manual(
        global_layer_indices=(0, 1),
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        num_hidden_layers=2,
    )
    assert topology.independent_kv_layers == (0, 1)
