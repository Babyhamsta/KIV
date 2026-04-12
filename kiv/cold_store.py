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

    One instance per independent KV layer (3 total: layers 4, 9, 14).
    Shared layers (19-34) resolve to layer 14's store.
    """

    def __init__(self, config: KIVConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self._num_kv_groups = config.num_query_heads // config.num_kv_heads
        self._page_size = config.page_size

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
        if config.prefetch_stream and device.type == "cuda":
            self._fetch_stream = torch.cuda.Stream(device)

    @property
    def cold_length(self) -> int:
        partial_len = self._partial_k.shape[2] if self._partial_k is not None else 0
        return self._num_finalized + partial_len

    @property
    def num_pages(self) -> int:
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
            self._pages_materialized = False  # invalidate cached concatenation

            # Trim partial buffer
            self._partial_k = self._partial_k[:, :, self._page_size:, :].contiguous()
            self._partial_v = self._partial_v[:, :, self._page_size:, :].contiguous()

    def score_and_fetch_cold(
        self, query: torch.Tensor, scaling: float, config: KIVConfig
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """
        Two-stage coarse-to-fine cold scoring + V fetch.

        Stage 1 (coarse): Score page summaries on GPU — select top pages.
        Stage 2 (fine): Fetch selected pages' K from CPU, exact score,
                        select top-P tokens, fetch their V vectors.

        Returns:
            (top_scores, V_fetched) — [B, H_q, Q, P] and [B, H_q, Q, P, D]
            or (None, None) if no cold tokens.
        """
        if self.cold_length == 0:
            return None, None

        self._materialize()

        B = query.shape[0]
        H_q = query.shape[1]
        Q = query.shape[2]
        D = query.shape[3]
        P = min(config.top_p, self.cold_length)

        # Expand query for MQA (already [B, 8, Q, D] from caller)
        # Page summaries are [B, 1, num_pages, D] — need expansion

        # ── Collect candidate K tokens for fine scoring ──
        candidate_k_parts = []    # list of [B, H_kv, num_tokens, D] tensors on GPU
        candidate_global_idx = []  # matching global cold indices for V fetch

        # Part A: Coarse pass over finalized pages (if any)
        if self.num_pages > 0:
            # Score summaries on GPU — no CPU transfer needed
            summaries_exp = _repeat_kv(self._page_summaries, self._num_kv_groups)
            # [B, 8, Q, num_pages]
            coarse_scores = torch.matmul(query, summaries_exp.transpose(-2, -1)) * scaling
            del summaries_exp

            # Select top pages
            num_select = min(config.top_pages, self.num_pages)
            _, top_page_idx = coarse_scores.topk(num_select, dim=-1)  # [B, 8, Q, num_select]
            del coarse_scores

            # Deduplicate page indices across heads for fetching
            # All heads index the same K pages (1 KV head), so flatten and unique
            flat_pages = top_page_idx.reshape(B, -1)  # [B, 8*Q*num_select]
            for b in range(B):
                unique_pages = flat_pages[b].unique().sort().values  # sorted page indices
                # Fetch those pages' K tokens from CPU → GPU
                page_k_parts = []
                page_global_indices = []
                for pg_idx in unique_pages.tolist():
                    start = pg_idx * self._page_size
                    end = start + self._page_size
                    k_page = self._k_pages[b:b+1, :, start:end, :].to(self.device, non_blocking=True)
                    page_k_parts.append(k_page)
                    page_global_indices.extend(range(start, end))

                if page_k_parts:
                    candidate_k_parts.append(torch.cat(page_k_parts, dim=2))
                    candidate_global_idx.append(page_global_indices)
                else:
                    candidate_k_parts.append(None)
                    candidate_global_idx.append([])

        else:
            for b in range(B):
                candidate_k_parts.append(None)
                candidate_global_idx.append([])

        # Part B: Include partial page tokens (already on GPU)
        partial_len = self._partial_len
        partial_global_start = self._num_finalized

        # ── Fine scoring per batch element ──
        # For simplicity with B=1 (inference), optimize for that case
        all_top_scores = []
        all_V_fetched = []

        for b in range(B):
            # Build candidate K tensor for this batch
            parts = []
            global_indices = list(candidate_global_idx[b])

            if candidate_k_parts[b] is not None:
                parts.append(candidate_k_parts[b])

            if partial_len > 0:
                parts.append(self._partial_k[b:b+1])
                global_indices.extend(range(partial_global_start, partial_global_start + partial_len))

            if not parts:
                # No candidates at all — shouldn't happen if cold_length > 0
                return None, None

            fine_k = torch.cat(parts, dim=2)  # [1, H_kv, num_candidates, D]
            fine_k_exp = _repeat_kv(fine_k, self._num_kv_groups)  # [1, 8, num_candidates, D]

            # Exact scoring
            fine_scores = torch.matmul(
                query[b:b+1], fine_k_exp.transpose(-2, -1)
            ) * scaling  # [1, 8, Q, num_candidates]

            # Select top-P (capped at number of candidates)
            actual_P = min(P, fine_scores.shape[-1])
            top_vals, top_local_idx = fine_scores.topk(actual_P, dim=-1)  # [1, 8, Q, actual_P]

            # Pad to P if needed (ensures consistent shape across batch elements)
            if actual_P < P:
                pad_size = P - actual_P
                top_vals = torch.nn.functional.pad(top_vals, (0, pad_size), value=float("-inf"))
                top_local_idx = torch.nn.functional.pad(top_local_idx, (0, pad_size), value=0)
            all_top_scores.append(top_vals)

            # Map local indices → global cold indices for V fetch
            idx_map = torch.tensor(global_indices, device=self.device, dtype=torch.long)
            global_top_idx = idx_map[top_local_idx.reshape(-1)].reshape(top_local_idx.shape)
            # [1, 8, Q, P] — global cold indices

            # Fetch V for these tokens
            V_b = self._fetch_v_for_indices(b, global_top_idx.squeeze(0), actual_P)
            all_V_fetched.append(V_b)

            del fine_k, fine_k_exp, fine_scores

        top_scores = torch.cat(all_top_scores, dim=0)  # [B, 8, Q, P]
        V_fetched = torch.cat(all_V_fetched, dim=0)    # [B, 8, Q, P, D]
        return top_scores, V_fetched

    def _fetch_v_for_indices(
        self, batch_idx: int, indices: torch.Tensor, P: int
    ) -> torch.Tensor:
        """
        Fetch V vectors for specific global cold indices via batched gather.

        indices: [H_q, Q, P] — global cold token indices
        Returns: [1, H_q, Q, P, D] on GPU
        """
        H_q, Q, _ = indices.shape
        D = self.config.head_dim

        flat_idx = indices.reshape(-1)  # [H_q*Q*P]
        unique_idx, inverse = flat_idx.unique(return_inverse=True)
        unique_cpu = unique_idx.cpu()

        # Separate finalized (CPU) and partial (GPU) indices
        finalized_mask = unique_cpu < self._num_finalized
        finalized_idx = unique_cpu[finalized_mask]
        partial_idx = unique_cpu[~finalized_mask] - self._num_finalized

        # Batched fetch from CPU pinned V store → GPU (single transfer)
        parts = []
        part_order = []  # tracks which unique positions these fill

        if finalized_idx.numel() > 0:
            v_batch_cpu = self._v_store[batch_idx, 0, finalized_idx, :]  # [N_fin, D]
            stream = self._fetch_stream or torch.cuda.current_stream(self.device)
            with torch.cuda.stream(stream):
                v_batch_gpu = v_batch_cpu.to(self.device, non_blocking=True)
            if self._fetch_stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self._fetch_stream)
            parts.append(v_batch_gpu)
            part_order.append(finalized_mask)

        # Batched gather from GPU partial buffer
        if partial_idx.numel() > 0:
            partial_idx_gpu = partial_idx.to(self.device)
            v_partial = self._partial_v[batch_idx, 0, partial_idx_gpu, :]  # [N_part, D]
            parts.append(v_partial)
            part_order.append(~finalized_mask)

        # Reconstruct in unique_idx order
        v_unique = torch.empty(unique_idx.shape[0], D, device=self.device, dtype=self._v_store.dtype if self._v_store is not None else torch.bfloat16)
        for tensor, mask in zip(parts, part_order):
            v_unique[mask.to(self.device)] = tensor

        v_expanded = v_unique[inverse]  # [H_q*Q*P, D]
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
