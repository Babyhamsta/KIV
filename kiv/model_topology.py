"""Model architecture descriptor for KIV."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelTopology:
    """Immutable descriptor of a model's architecture relevant to KIV.

    Captures which layers use global attention, which share KV,
    and the head geometry. Either auto-detected via ``detect_topology``
    or constructed manually for unsupported models.

    Invariants (enforced in ``__post_init__``):

    * ``independent_kv_layers`` is a subset of ``global_layer_indices``.
    * ``kv_sharing_map`` keys are all members of ``global_layer_indices``
      (a shared global layer must still be declared as a global layer
      so the KIV attention wrapper binds its cache).
    * ``kv_sharing_map`` values are all members of
      ``independent_kv_layers`` (the source of a share must own a cold
      store).
    """

    model_family: str
    num_hidden_layers: int
    global_layer_indices: tuple[int, ...]
    independent_kv_layers: tuple[int, ...]
    kv_sharing_map: dict[int, int] = field(default_factory=dict)
    num_query_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0

    def __post_init__(self) -> None:
        globals_set = set(self.global_layer_indices)
        independent_set = set(self.independent_kv_layers)

        missing_independent = independent_set - globals_set
        if missing_independent:
            raise ValueError(
                "independent_kv_layers must be a subset of "
                f"global_layer_indices; extras: {sorted(missing_independent)}"
            )

        missing_keys = set(self.kv_sharing_map.keys()) - globals_set
        if missing_keys:
            raise ValueError(
                "kv_sharing_map keys must all appear in "
                f"global_layer_indices; extras: {sorted(missing_keys)}"
            )

        missing_sources = set(self.kv_sharing_map.values()) - independent_set
        if missing_sources:
            raise ValueError(
                "kv_sharing_map values must all appear in "
                f"independent_kv_layers; extras: {sorted(missing_sources)}"
            )

    @staticmethod
    def manual(
        *,
        global_layer_indices: tuple[int, ...],
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_hidden_layers: int,
        independent_kv_layers: tuple[int, ...] | None = None,
        kv_sharing_map: dict[int, int] | None = None,
        model_family: str = "unknown",
    ) -> ModelTopology:
        """Construct a ModelTopology manually for unsupported models."""
        if independent_kv_layers is None:
            independent_kv_layers = global_layer_indices
        return ModelTopology(
            model_family=model_family,
            num_hidden_layers=num_hidden_layers,
            global_layer_indices=global_layer_indices,
            independent_kv_layers=independent_kv_layers,
            kv_sharing_map=kv_sharing_map or {},
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
