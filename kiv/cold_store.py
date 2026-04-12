"""
Cold KV storage with page-based coarse-to-fine retrieval.

Instead of scoring all cold K vectors every decode step (O(cold_len) CPU→GPU
transfer), we maintain page summaries on GPU and do a two-stage lookup:
  1. Coarse: Q @ page_summaries.T  (GPU only, trivially fast)
  2. Fine:   Fetch top-32 pages' K from CPU, exact score, select top-P

Optimizations:
  - Batched page fetch: one CPU→GPU transfer for all selected pages (K+V together)
  - Candidate caching: shared layers reuse coarse+fetch results, only re-run fine scoring
  - Vectorized indexing: all index operations stay on GPU, no .tolist() or Python loops
"""

from __future__ import annotations

import torch

from .config import KIVConfig
from .model_topology import ModelTopology


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for MQA/GQA. [B, H_kv, T, D] -> [B, H_kv*n_rep, T, D]."""
    if n_rep == 1:
        return hidden_states
    B, H, T, D = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(B, H, n_rep, T, D)
        .reshape(B, H * n_rep, T, D)
    )


class ColdKVStore:
    """
    Manages cold (evicted) tokens with page-based coarse-to-fine retrieval.

    Tokens are grouped into fixed-size pages. Each page has a summary vector
    (mean of its K vectors) stored on GPU for fast coarse scoring. The actual
    per-token K and V vectors stay on CPU pinned memory and are fetched only
    for the top pages selected by the coarse pass.

    One instance per independent KV layer.
    """

    def __init__(
        self,
        kiv_config: KIVConfig,
        topology: ModelTopology,
        device: torch.device,
    ) -> None:
        self.kiv_config = kiv_config
        self.device = device
        self._num_kv_groups = topology.num_query_heads // topology.num_kv_heads
        self._num_kv_heads = topology.num_kv_heads
        self._head_dim = topology.head_dim
        self._page_size = kiv_config.page_size

        # Page summaries on GPU — mean K per finalized page
        self._summary_list: list[torch.Tensor] = []
        self._page_summaries: torch.Tensor | None = None

        # Partial page buffers on GPU — tokens not yet forming a full page
        self._partial_k: torch.Tensor | None = None
        self._partial_v: torch.Tensor | None = None

        # Finalized token storage on CPU pinned memory
        self._k_page_list: list[torch.Tensor] = []
        self._v_page_list: list[torch.Tensor] = []
        self._k_pages: torch.Tensor | None = None
        self._v_store: torch.Tensor | None = None
        self._num_finalized: int = 0
        self._pages_materialized: bool = False

        # Async fetch stream
        self._fetch_stream: torch.cuda.Stream | None = None
        if kiv_config.prefetch_stream and device.type == "cuda":
            self._fetch_stream = torch.cuda.Stream(device)

        # Candidate cache for shared-layer reuse
        self._cached_candidate_k: torch.Tensor | None = None
        self._cached_candidate_v: torch.Tensor | None = None
        self._cached_candidate_idx: torch.Tensor | None = None
        self._cached_step: int = -1

    def _cpu_storage_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy to CPU storage, pinning only when CUDA fetches will use it."""
        cpu_tensor = tensor.cpu()
        if self.device.type == "cuda" and torch.cuda.is_available():
            return cpu_tensor.pin_memory()
        return cpu_tensor

    def _copy_to_store_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move fetched CPU storage back to the store device."""
        if self.device.type != "cuda":
            return tensor.to(self.device)

        stream = self._fetch_stream or torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            result = tensor.to(self.device, non_blocking=True)
        if self._fetch_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._fetch_stream)
        return result

    @property
    def cold_length(self) -> int:
        partial_len = self._partial_k.shape[2] if self._partial_k is not None else 0
        return self._num_finalized + partial_len

    @property
    def num_pages(self) -> int:
        if self._pages_materialized and self._page_summaries is not None:
            return self._page_summaries.shape[2]
        return len(self._summary_list)

    def _materialize(self) -> None:
        """Concatenate page lists into tensors. Called once before first scoring."""
        if self._pages_materialized:
            return
        if self._summary_list:
            self._page_summaries = torch.cat(self._summary_list, dim=2)
        else:
            self._page_summaries = None
        if self._k_page_list:
            self._k_pages = torch.cat(self._k_page_list, dim=2)
            self._v_store = torch.cat(self._v_page_list, dim=2)
        else:
            self._k_pages = None
            self._v_store = None
        self._pages_materialized = True

    @property
    def _partial_len(self) -> int:
        return self._partial_k.shape[2] if self._partial_k is not None else 0

    def evict_from_hot(
        self, k_evicted: torch.Tensor, v_evicted: torch.Tensor
    ) -> None:
        """
        Append evicted tokens to cold store, building pages as they fill.
        """
        self._cached_candidate_k = None
        self._cached_candidate_v = None
        self._cached_candidate_idx = None
        self._cached_step = -1

        if self._partial_k is None:
            self._partial_k = k_evicted
            self._partial_v = v_evicted
        else:
            self._partial_k = torch.cat([self._partial_k, k_evicted], dim=2)
            self._partial_v = torch.cat([self._partial_v, v_evicted], dim=2)

        while self._partial_len >= self._page_size:
            page_k = self._partial_k[:, :, :self._page_size, :]
            page_v = self._partial_v[:, :, :self._page_size, :]

            summary = page_k.mean(dim=2, keepdim=True)
            self._summary_list.append(summary)

            k_cpu = self._cpu_storage_copy(page_k)
            v_cpu = self._cpu_storage_copy(page_v)
            self._k_page_list.append(k_cpu)
            self._v_page_list.append(v_cpu)

            self._num_finalized += self._page_size
            if self._pages_materialized and self._k_pages is not None:
                self._k_pages = torch.cat([self._k_pages, k_cpu], dim=2)
                self._v_store = torch.cat([self._v_store, v_cpu], dim=2)
                self._page_summaries = torch.cat([self._page_summaries, summary], dim=2)
                self._summary_list.clear()
                self._k_page_list.clear()
                self._v_page_list.clear()
            else:
                self._pages_materialized = False

            self._partial_k = self._partial_k[:, :, self._page_size:, :].contiguous()
            self._partial_v = self._partial_v[:, :, self._page_size:, :].contiguous()

    # ── Vectorized candidate selection ──

    def _select_candidates(
        self, query: torch.Tensor, scaling: float, kiv_config: KIVConfig
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Coarse page scoring + batched K/V fetch. All GPU, no Python loops.

        Returns (candidate_k, candidate_v, candidate_indices) as GPU tensors,
        or (None, None, None) if no candidates.
        candidate_k/v: [B, H_kv, num_candidates, D]
        candidate_indices: [num_candidates] long tensor on GPU
        """
        self._materialize()

        finalized_k = None
        finalized_v = None
        finalized_indices = None

        if self.num_pages > 0:
            # Coarse scoring — GPU matmul on page summaries
            summaries_exp = _repeat_kv(self._page_summaries, self._num_kv_groups)
            coarse_scores = torch.matmul(
                query, summaries_exp.transpose(-2, -1)
            ) * scaling
            del summaries_exp

            num_select = min(kiv_config.top_pages, self.num_pages)
            _, top_page_idx = coarse_scores.topk(num_select, dim=-1)
            del coarse_scores

            # Deduplicate pages across batches, heads, and query positions.
            unique_pages = top_page_idx.reshape(-1).unique(sorted=True)

            # Build token indices for all selected pages via broadcast — no Python loop
            page_starts = unique_pages * self._page_size
            offsets = torch.arange(self._page_size, device=self.device)
            # [num_unique, page_size] → flatten
            all_token_idx = (page_starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)

            # Single batched CPU→GPU transfer for K and V together
            all_token_idx_cpu = all_token_idx.cpu()
            k_cpu = self._k_pages[:, :, all_token_idx_cpu, :]
            v_cpu = self._v_store[:, :, all_token_idx_cpu, :]

            finalized_k = self._copy_to_store_device(k_cpu)
            finalized_v = self._copy_to_store_device(v_cpu)

            finalized_indices = all_token_idx  # already on GPU

        # Append partial tokens (already on GPU)
        parts_k = []
        parts_v = []
        parts_idx = []

        if finalized_k is not None:
            parts_k.append(finalized_k)
            parts_v.append(finalized_v)
            parts_idx.append(finalized_indices)

        partial_len = self._partial_len
        if partial_len > 0:
            parts_k.append(self._partial_k)
            parts_v.append(self._partial_v)
            parts_idx.append(
                torch.arange(
                    self._num_finalized,
                    self._num_finalized + partial_len,
                    device=self.device,
                )
            )

        if not parts_k:
            return None, None, None

        candidate_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        candidate_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
        candidate_idx = torch.cat(parts_idx) if len(parts_idx) > 1 else parts_idx[0]

        return candidate_k, candidate_v, candidate_idx

    # ─�� Primary retrieval: returns K and V for concatenation ──

    def retrieve_top_kv(
        self,
        query: torch.Tensor,
        scaling: float,
        kiv_config: KIVConfig,
        step: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Coarse-to-fine retrieval returning top-P K and V for concatenation.

        If step matches cached step, reuses candidates from a prior call
        (skips coarse scoring + CPU fetch). This saves ~11ms per shared layer.

        Returns (K, V) both [B, H_kv, P, D], or (None, None).
        """
        if self.cold_length == 0:
            return None, None

        P = min(kiv_config.top_p, self.cold_length)

        # Reuse cached candidates if same decode step (shared layer optimization)
        if (
            step >= 0
            and step == self._cached_step
            and self._cached_candidate_k is not None
            and self._cached_candidate_k.shape[0] == query.shape[0]
        ):
            candidate_k = self._cached_candidate_k
            candidate_v = self._cached_candidate_v
            candidate_idx = self._cached_candidate_idx
        else:
            candidate_k, candidate_v, candidate_idx = self._select_candidates(
                query, scaling, kiv_config
            )
            if candidate_k is None:
                return None, None
            # Cache for shared layers
            self._cached_candidate_k = candidate_k
            self._cached_candidate_v = candidate_v
            self._cached_candidate_idx = candidate_idx
            if step >= 0:
                self._cached_step = step

        # Fine scoring — always runs (query differs per layer)
        fine_k_exp = _repeat_kv(candidate_k, self._num_kv_groups)
        fine_scores = torch.matmul(
            query, fine_k_exp.transpose(-2, -1)
        ) * scaling

        # Aggregate across heads + Q positions
        agg_scores = fine_scores.amax(dim=(1, 2))

        actual_P = min(P, agg_scores.shape[-1])
        _, top_local_idx = agg_scores.topk(actual_P, dim=-1)

        # Select K and V via GPU index — no CPU round-trip
        gather_idx = top_local_idx[:, None, :, None].expand(
            -1, candidate_k.shape[1], -1, candidate_k.shape[3]
        )
        k_out = torch.gather(candidate_k, dim=2, index=gather_idx)
        v_out = torch.gather(candidate_v, dim=2, index=gather_idx)

        # Pad to P for consistent shape
        if k_out.shape[2] < P:
            pad_size = P - k_out.shape[2]
            k_out = torch.nn.functional.pad(k_out, (0, 0, 0, pad_size))
            v_out = torch.nn.functional.pad(v_out, (0, 0, 0, pad_size))

        return k_out, v_out

    # ── Legacy method (kept for backward compat) ──

    def score_and_fetch_cold(
        self, query: torch.Tensor, scaling: float, config: KIVConfig
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Deprecated: use retrieve_top_kv() instead."""
        if self.cold_length == 0:
            return None, None

        self._materialize()

        B = query.shape[0]
        H_q = query.shape[1]
        Q = query.shape[2]
        P = min(config.top_p, self.cold_length)

        candidate_k, _, candidate_idx = self._select_candidates(
            query, scaling, config
        )
        if candidate_k is None:
            return None, None

        all_top_scores = []
        all_V_fetched = []

        for b in range(B):
            fine_k_exp = _repeat_kv(candidate_k[b:b+1], self._num_kv_groups)
            fine_scores = torch.matmul(
                query[b:b+1], fine_k_exp.transpose(-2, -1)
            ) * scaling

            actual_P = min(P, fine_scores.shape[-1])
            top_vals, top_local_idx = fine_scores.topk(actual_P, dim=-1)

            if actual_P < P:
                pad_size = P - actual_P
                top_vals = torch.nn.functional.pad(
                    top_vals, (0, pad_size), value=float("-inf")
                )
                top_local_idx = torch.nn.functional.pad(
                    top_local_idx, (0, pad_size), value=0
                )
            all_top_scores.append(top_vals)

            # Map to global indices — GPU tensor indexing
            global_top_idx = candidate_idx[top_local_idx.reshape(-1)].reshape(
                top_local_idx.shape
            )
            V_b = self._fetch_v_for_indices_legacy(
                b, global_top_idx.squeeze(0), P
            )
            all_V_fetched.append(V_b)

            del fine_k_exp, fine_scores

        top_scores = torch.cat(all_top_scores, dim=0)
        V_fetched = torch.cat(all_V_fetched, dim=0)
        return top_scores, V_fetched

    def _fetch_v_for_indices_legacy(
        self, batch_idx: int, indices: torch.Tensor, P: int
    ) -> torch.Tensor:
        """Legacy V fetch returning [1, H_q, Q, P, D] with head expansion."""
        H_q, Q, _ = indices.shape
        D = self._head_dim

        flat_idx = indices.reshape(-1)
        unique_idx, inverse = flat_idx.unique(return_inverse=True)
        unique_cpu = unique_idx.cpu()

        finalized_mask = unique_cpu < self._num_finalized
        finalized_idx = unique_cpu[finalized_mask]
        partial_idx = unique_cpu[~finalized_mask] - self._num_finalized

        parts = []
        part_order = []

        if finalized_idx.numel() > 0:
            v_batch_cpu = self._v_store[batch_idx, 0, finalized_idx, :]
            v_batch_gpu = self._copy_to_store_device(v_batch_cpu)
            parts.append(v_batch_gpu)
            part_order.append(finalized_mask)

        if partial_idx.numel() > 0:
            partial_idx_gpu = partial_idx.to(self.device)
            v_partial = self._partial_v[batch_idx, 0, partial_idx_gpu, :]
            parts.append(v_partial)
            part_order.append(~finalized_mask)

        dtype = (self._v_store.dtype if self._v_store is not None
                 else (self._partial_v.dtype if self._partial_v is not None
                       else torch.bfloat16))
        v_unique = torch.empty(
            unique_idx.shape[0], D, device=self.device, dtype=dtype
        )
        for tensor, mask in zip(parts, part_order):
            v_unique[mask.to(self.device)] = tensor

        v_expanded = v_unique[inverse]
        return v_expanded.view(1, H_q, Q, P, D)

    def reset(self) -> None:
        """Clear all cold storage."""
        self._summary_list.clear()
        self._page_summaries = None
        self._k_page_list.clear()
        self._v_page_list.clear()
        self._k_pages = None
        self._v_store = None
        self._partial_k = None
        self._partial_v = None
        self._num_finalized = 0
        self._pages_materialized = False
        self._cached_candidate_k = None
        self._cached_candidate_v = None
        self._cached_candidate_idx = None
        self._cached_step = -1

    def memory_bytes(self) -> dict[str, int]:
        """Report memory usage by tier."""
        k_cpu = self._k_pages.nelement() * self._k_pages.element_size() if self._k_pages is not None else 0
        v_cpu = self._v_store.nelement() * self._v_store.element_size() if self._v_store is not None else 0
        summaries_gpu = self._page_summaries.nelement() * self._page_summaries.element_size() if self._page_summaries is not None else 0
        partial_gpu = 0
        if self._partial_k is not None:
            partial_gpu += self._partial_k.nelement() * self._partial_k.element_size()
        if self._partial_v is not None:
            partial_gpu += self._partial_v.nelement() * self._partial_v.element_size()
        return {
            "k_index_cpu": k_cpu,
            "v_store_cpu": v_cpu,
            "page_summaries_gpu": summaries_gpu,
            "partial_gpu": partial_gpu,
        }
