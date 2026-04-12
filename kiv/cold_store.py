"""
Cold KV storage with page-based coarse-to-fine retrieval.

Instead of scoring all cold K vectors every decode step (O(cold_len) CPU→GPU
transfer), we maintain page summaries on GPU and do a two-stage lookup:
  1. Coarse: Q @ page_summaries.T  (GPU only, trivially fast)
  2. Fine:   Fetch top-32 pages' K from CPU, exact score, select top-P

At 1M tokens this reduces per-layer cold scoring from ~75ms to ~4ms.
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
        self._summary_list: list[torch.Tensor] = []    # list of [B, H_kv, 1, D]
        self._page_summaries: torch.Tensor | None = None  # materialized on first score

        # Partial page buffers on GPU — tokens not yet forming a full page
        self._partial_k: torch.Tensor | None = None  # [B, H_kv, partial_len, D]
        self._partial_v: torch.Tensor | None = None  # [B, H_kv, partial_len, D]

        # Finalized token storage on CPU pinned memory — stored as page-sized chunks
        self._k_page_list: list[torch.Tensor] = []    # list of [B, H_kv, page_size, D]
        self._v_page_list: list[torch.Tensor] = []    # list of [B, H_kv, page_size, D]
        self._k_pages: torch.Tensor | None = None     # materialized on first score
        self._v_store: torch.Tensor | None = None      # materialized on first score
        self._num_finalized: int = 0
        self._pages_materialized: bool = False

        # Async fetch stream
        self._fetch_stream: torch.cuda.Stream | None = None
        if kiv_config.prefetch_stream and device.type == "cuda":
            self._fetch_stream = torch.cuda.Stream(device)

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

        New tokens go into the partial buffer (GPU). When the partial buffer
        reaches page_size, we finalize a page: compute the summary (mean K),
        store it on GPU, and move the token-level K/V to CPU pinned memory.
        """
        # Append to partial buffer (stays on GPU)
        if self._partial_k is None:
            self._partial_k = k_evicted
            self._partial_v = v_evicted
        else:
            self._partial_k = torch.cat([self._partial_k, k_evicted], dim=2)
            self._partial_v = torch.cat([self._partial_v, v_evicted], dim=2)

        # Finalize full pages
        while self._partial_len >= self._page_size:
            page_k = self._partial_k[:, :, :self._page_size, :]  # [B, H_kv, page_size, D]
            page_v = self._partial_v[:, :, :self._page_size, :]

            # Page summary = mean K vector (stays on GPU). Append to list — O(1).
            summary = page_k.mean(dim=2, keepdim=True)  # [B, H_kv, 1, D]
            self._summary_list.append(summary)

            # Move page tokens to CPU pinned memory. Append to list — O(1).
            k_cpu = page_k.cpu().pin_memory()
            v_cpu = page_v.cpu().pin_memory()
            self._k_page_list.append(k_cpu)
            self._v_page_list.append(v_cpu)

            self._num_finalized += self._page_size
            if self._pages_materialized and self._k_pages is not None:
                self._k_pages = torch.cat([self._k_pages, k_cpu], dim=2)
                self._v_store = torch.cat([self._v_store, v_cpu], dim=2)
                self._page_summaries = torch.cat([self._page_summaries, summary], dim=2)
                # Lists served as source for initial materialization.
                # Now materialized tensors are authoritative — free the duplicates.
                self._summary_list.clear()
                self._k_page_list.clear()
                self._v_page_list.clear()
            else:
                self._pages_materialized = False

            # Trim partial buffer
            self._partial_k = self._partial_k[:, :, self._page_size:, :].contiguous()
            self._partial_v = self._partial_v[:, :, self._page_size:, :].contiguous()

    # ── Coarse-to-fine candidate selection (shared by both retrieval methods) ──

    def _select_candidates(
        self, query: torch.Tensor, scaling: float, kiv_config: KIVConfig
    ) -> list[tuple[torch.Tensor | None, list[int]]]:
        """
        Run coarse page scoring and collect candidate K tensors per batch element.

        Returns list of (candidate_k, global_indices) per batch element.
        candidate_k includes both page-selected and partial tokens.
        """
        self._materialize()

        B = query.shape[0]
        candidate_k_parts: list[torch.Tensor | None] = []
        candidate_global_idx: list[list[int]] = []

        # Part A: Coarse pass over finalized pages
        if self.num_pages > 0:
            summaries_exp = _repeat_kv(self._page_summaries, self._num_kv_groups)
            coarse_scores = torch.matmul(query, summaries_exp.transpose(-2, -1)) * scaling
            del summaries_exp

            num_select = min(kiv_config.top_pages, self.num_pages)
            _, top_page_idx = coarse_scores.topk(num_select, dim=-1)
            del coarse_scores

            flat_pages = top_page_idx.reshape(B, -1)
            for b in range(B):
                unique_pages = flat_pages[b].unique().sort().values
                page_k_parts = []
                page_global_indices: list[int] = []
                for pg_idx in unique_pages.tolist():
                    start = pg_idx * self._page_size
                    end = start + self._page_size
                    k_page = self._k_pages[b:b+1, :, start:end, :].to(
                        self.device, non_blocking=True
                    )
                    page_k_parts.append(k_page)
                    page_global_indices.extend(range(start, end))

                if page_k_parts:
                    candidate_k_parts.append(torch.cat(page_k_parts, dim=2))
                    candidate_global_idx.append(page_global_indices)
                else:
                    candidate_k_parts.append(None)
                    candidate_global_idx.append([])
        else:
            for _ in range(B):
                candidate_k_parts.append(None)
                candidate_global_idx.append([])

        # Part B: Include partial page tokens (already on GPU)
        partial_len = self._partial_len
        partial_global_start = self._num_finalized

        results = []
        for b in range(B):
            parts = []
            global_indices = list(candidate_global_idx[b])

            if candidate_k_parts[b] is not None:
                parts.append(candidate_k_parts[b])

            if partial_len > 0:
                parts.append(self._partial_k[b:b+1])
                global_indices.extend(
                    range(partial_global_start, partial_global_start + partial_len)
                )

            if parts:
                fine_k = torch.cat(parts, dim=2)
            else:
                fine_k = None

            results.append((fine_k, global_indices))

        return results

    # ── Primary retrieval method: returns K and V for concatenation ──

    def retrieve_top_kv(
        self, query: torch.Tensor, scaling: float, kiv_config: KIVConfig
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Coarse-to-fine retrieval returning top-P K and V in natural KV shape.

        Scores are aggregated across query heads (max-pool) to select a
        single set of P tokens shared by all heads. Returns K and V in
        [B, H_kv, P, D] shape suitable for concatenation with hot K/V.

        Args:
            query:      [B, H_q, Q, D]
            scaling:    attention scaling factor
            kiv_config: KIVConfig with top_p, top_pages

        Returns:
            (K_selected, V_selected) both [B, H_kv, P, D], or (None, None).
        """
        if self.cold_length == 0:
            return None, None

        B = query.shape[0]
        P = min(kiv_config.top_p, self.cold_length)

        candidates = self._select_candidates(query, scaling, kiv_config)

        all_K = []
        all_V = []

        for b in range(B):
            fine_k, global_indices = candidates[b]
            if fine_k is None or not global_indices:
                dtype = (self._v_store.dtype if self._v_store is not None
                         else (self._partial_v.dtype if self._partial_v is not None
                               else torch.bfloat16))
                k_out = torch.zeros(1, self._num_kv_heads, P, self._head_dim,
                                    device=self.device, dtype=dtype)
                v_out = torch.zeros(1, self._num_kv_heads, P, self._head_dim,
                                    device=self.device, dtype=dtype)
                all_K.append(k_out)
                all_V.append(v_out)
                continue

            fine_k_exp = _repeat_kv(fine_k, self._num_kv_groups)

            # Score: [1, H_q, Q, num_candidates]
            fine_scores = torch.matmul(
                query[b:b+1], fine_k_exp.transpose(-2, -1)
            ) * scaling

            # Aggregate across query heads: max-pool → [1, 1, Q, num_candidates]
            agg_scores = fine_scores.max(dim=1, keepdim=True).values
            # Flatten Q dim for selection: [1, 1, Q*num_candidates] effectively
            # For decode (Q=1), agg_scores is [1, 1, 1, num_candidates]
            # Just select from the last dim
            agg_flat = agg_scores.squeeze(1).squeeze(0)  # [Q, num_candidates]
            # Max across Q positions too → [num_candidates]
            if agg_flat.ndim > 1:
                agg_flat = agg_flat.max(dim=0).values

            actual_P = min(P, agg_flat.shape[0])
            _, top_local_idx = agg_flat.topk(actual_P)  # [actual_P]

            # Map to global cold indices
            idx_map = torch.tensor(global_indices, device=self.device, dtype=torch.long)
            global_selected = idx_map[top_local_idx]

            # Fetch K and V for selected global indices
            k_out, v_out = self._fetch_kv_for_indices(b, global_selected)
            # k_out, v_out: [1, H_kv, actual_P, D]

            # Pad to P for consistent shape across batch elements
            if k_out.shape[2] < P:
                pad_size = P - k_out.shape[2]
                k_out = torch.nn.functional.pad(k_out, (0, 0, 0, pad_size))
                v_out = torch.nn.functional.pad(v_out, (0, 0, 0, pad_size))

            all_K.append(k_out)
            all_V.append(v_out)

            del fine_k, fine_k_exp, fine_scores

        K_selected = torch.cat(all_K, dim=0)  # [B, H_kv, P, D]
        V_selected = torch.cat(all_V, dim=0)  # [B, H_kv, P, D]
        return K_selected, V_selected

    def _fetch_kv_for_indices(
        self, batch_idx: int, global_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch K and V vectors for specific global cold indices.

        global_indices: [N] — 1D tensor of global cold token indices
        Returns: (K, V) each [1, H_kv, N, D] on GPU
        """
        H_kv = self._num_kv_heads
        D = self._head_dim

        unique_cpu = global_indices.cpu()

        # Separate finalized (CPU) and partial (GPU) indices
        finalized_mask = unique_cpu < self._num_finalized
        finalized_idx = unique_cpu[finalized_mask]
        partial_idx = unique_cpu[~finalized_mask] - self._num_finalized

        k_parts = []
        v_parts = []
        order = []

        if finalized_idx.numel() > 0:
            # Fetch from CPU pinned memory
            k_fin = self._k_pages[batch_idx:batch_idx+1, :, finalized_idx, :]
            v_fin = self._v_store[batch_idx:batch_idx+1, :, finalized_idx, :]
            stream = self._fetch_stream or torch.cuda.current_stream(self.device)
            with torch.cuda.stream(stream):
                k_fin_gpu = k_fin.to(self.device, non_blocking=True)
                v_fin_gpu = v_fin.to(self.device, non_blocking=True)
            if self._fetch_stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self._fetch_stream)
            k_parts.append(k_fin_gpu)
            v_parts.append(v_fin_gpu)
            order.append(finalized_mask)

        if partial_idx.numel() > 0:
            partial_idx_gpu = partial_idx.to(self.device)
            k_part = self._partial_k[batch_idx:batch_idx+1, :, partial_idx_gpu, :]
            v_part = self._partial_v[batch_idx:batch_idx+1, :, partial_idx_gpu, :]
            k_parts.append(k_part)
            v_parts.append(v_part)
            order.append(~finalized_mask)

        N = global_indices.shape[0]
        dtype = (self._v_store.dtype if self._v_store is not None
                else (self._partial_v.dtype if self._partial_v is not None
                      else torch.bfloat16))
        k_out = torch.empty(1, H_kv, N, D, device=self.device, dtype=dtype)
        v_out = torch.empty(1, H_kv, N, D, device=self.device, dtype=dtype)

        for k_t, v_t, mask in zip(k_parts, v_parts, order):
            mask_gpu = mask.to(self.device)
            k_out[0, :, mask_gpu, :] = k_t[0]
            v_out[0, :, mask_gpu, :] = v_t[0]

        return k_out, v_out

    # ── Legacy method (kept for backward compat during transition) ──

    def score_and_fetch_cold(
        self, query: torch.Tensor, scaling: float, config: KIVConfig
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Legacy two-stage scoring returning (scores, V) for two-partition attention.

        Deprecated: use retrieve_top_kv() instead.
        """
        if self.cold_length == 0:
            return None, None

        self._materialize()

        B = query.shape[0]
        H_q = query.shape[1]
        Q = query.shape[2]
        D = query.shape[3]
        P = min(config.top_p, self.cold_length)

        candidates = self._select_candidates(query, scaling, config)

        all_top_scores = []
        all_V_fetched = []

        for b in range(B):
            fine_k, global_indices = candidates[b]
            if fine_k is None or not global_indices:
                return None, None

            fine_k_exp = _repeat_kv(fine_k, self._num_kv_groups)

            fine_scores = torch.matmul(
                query[b:b+1], fine_k_exp.transpose(-2, -1)
            ) * scaling

            actual_P = min(P, fine_scores.shape[-1])
            top_vals, top_local_idx = fine_scores.topk(actual_P, dim=-1)

            if actual_P < P:
                pad_size = P - actual_P
                top_vals = torch.nn.functional.pad(top_vals, (0, pad_size), value=float("-inf"))
                top_local_idx = torch.nn.functional.pad(top_local_idx, (0, pad_size), value=0)
            all_top_scores.append(top_vals)

            idx_map = torch.tensor(global_indices, device=self.device, dtype=torch.long)
            global_top_idx = idx_map[top_local_idx.reshape(-1)].reshape(top_local_idx.shape)

            # Pass P (padded size) since indices were padded to P
            V_b = self._fetch_v_for_indices_legacy(b, global_top_idx.squeeze(0), P)
            all_V_fetched.append(V_b)

            del fine_k, fine_k_exp, fine_scores

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
            stream = self._fetch_stream or torch.cuda.current_stream(self.device)
            with torch.cuda.stream(stream):
                v_batch_gpu = v_batch_cpu.to(self.device, non_blocking=True)
            if self._fetch_stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self._fetch_stream)
            parts.append(v_batch_gpu)
            part_order.append(finalized_mask)

        if partial_idx.numel() > 0:
            partial_idx_gpu = partial_idx.to(self.device)
            v_partial = self._partial_v[batch_idx, 0, partial_idx_gpu, :]
            parts.append(v_partial)
            part_order.append(~finalized_mask)

        dtype = self._v_store.dtype if self._v_store is not None else torch.bfloat16
        v_unique = torch.empty(unique_idx.shape[0], D, device=self.device, dtype=dtype)
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
