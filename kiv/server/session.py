"""Multi-slot session pool with prefix-reuse across parallel conversations.

KIV's value lives in keeping the KV cache warm across turns. Ollama's
HTTP protocol is stateless: each request carries the full conversation.
We bridge the two by comparing the incoming token stream against
per-slot token histories and reusing the longest matching prefix.

Why multiple slots: Open WebUI (and other common clients) fire auxiliary
``/api/chat`` calls for title generation, tag generation, follow-up
suggestions, and retrieval query rewriting. These look nothing like the
main conversation's token stream. A single-slot session would evict the
warm main-chat cache every time one of these aux calls arrived,
forcing a full re-prefill on the user's next real turn. Holding a
small pool of caches keyed by prefix handles this without user
configuration.

Design summary:

  * Pool holds up to ``max_slots`` caches. VRAM cost per slot is tiny
    (page summaries + partial page). CPU RAM is the real budget, but
    aux calls have short prompts so their slots stay small. Only the
    user's main chat slot grows to long contexts.
  * On each request, find the slot whose stored tokens are a full
    prefix of the request (append reuse). Longest such match wins.
  * If no slot matches as a prefix and an empty slot is available, fill
    it. Otherwise evict LRU.
  * The slot chosen is carried on the :class:`PrefillPlan` so the
    server can activate its cache on the model's attention modules
    before running prefill / decode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SLOTS = 8


@dataclass
class CacheSlot:
    """One live KV cache plus the tokens it currently holds."""

    cache: Any
    tokens: list[int] = field(default_factory=list)
    last_used: int = 0
    slot_id: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"CacheSlot(id={self.slot_id}, tokens={len(self.tokens)}, last_used={self.last_used})"


@dataclass
class PrefillPlan:
    """Instruction for a single request's prefill phase."""

    tail_tokens: list[int]
    reset: bool
    reused_prefix: int
    slot: CacheSlot


@dataclass
class KIVSession:
    """Pool of caches keyed by the token stream they were warmed with.

    The server owns exactly one ``KIVSession``. Concurrent requests
    serialize through an asyncio lock in ``app.py``; this class does
    not do its own locking.
    """

    middleware: Any
    max_slots: int = _DEFAULT_MAX_SLOTS
    slots: list[CacheSlot] = field(default_factory=list)
    _counter: int = 0
    _next_slot_id: int = 0

    def plan_request(self, request_tokens: list[int]) -> PrefillPlan:
        """Decide how to service a new request against the slot pool.

        Three resolution paths are tried in order:

        1. **Append reuse**: the slot's stored tokens are a strict
           prefix of the request. Reuse the cache, prefill only the
           tail. This is the common chat-append case.
        2. **Truncate reuse**: the request is a strict prefix of the
           slot's stored tokens (the slot has extra tokens the client
           is asking to regenerate). The slot's KV cache is rolled
           back in place to the request length minus one, and the
           final token is fed through as a one-token prefill. This is
           the chat-regenerate case; the truncation may fail if it
           would require dropping tokens that have already been
           evicted to cold, in which case the path is abandoned.
        3. **Fresh slot**: nothing usable; allocate (or LRU-evict) a
           slot and treat the request as a cold start.

        Whichever path saves the most prefix work wins.
        """
        if not request_tokens:
            raise ValueError("request_tokens must be non-empty")

        self._counter += 1

        best_append: CacheSlot | None = None
        best_append_prefix = 0
        best_truncate: CacheSlot | None = None
        best_truncate_prefix = 0

        for slot in self.slots:
            slot_len = len(slot.tokens)
            if slot_len == 0:
                continue
            prefix = _common_prefix_len(slot.tokens, request_tokens)
            if (
                slot_len < len(request_tokens)
                and prefix == slot_len
                and prefix > best_append_prefix
            ):
                best_append = slot
                best_append_prefix = prefix
                continue
            if (
                slot_len >= len(request_tokens)
                and prefix == len(request_tokens)
                and prefix > best_truncate_prefix
            ):
                best_truncate = slot
                best_truncate_prefix = prefix

        # Prefer truncate reuse when it offers strictly more shared
        # context than append. If the truncate request fails (would
        # need to drop tokens already evicted to cold), fall back to
        # the append candidate before giving up - a cold start with
        # 0 reuse is always worse than appending to a warm prefix.
        if (
            best_truncate is not None
            and best_truncate_prefix > max(best_append_prefix, 1)
        ):
            reuse_up_to = best_truncate_prefix - 1
            truncate_fn = getattr(best_truncate.cache, "truncate_to", None)
            if callable(truncate_fn) and bool(truncate_fn(reuse_up_to)):
                best_truncate.tokens = list(request_tokens[:reuse_up_to])
                best_truncate.last_used = self._counter
                return PrefillPlan(
                    tail_tokens=[request_tokens[-1]],
                    reset=False,
                    reused_prefix=reuse_up_to,
                    slot=best_truncate,
                )

        if best_append is not None and best_append_prefix > 0:
            best_append.last_used = self._counter
            tail = list(request_tokens[best_append_prefix:])
            return PrefillPlan(
                tail_tokens=tail,
                reset=False,
                reused_prefix=best_append_prefix,
                slot=best_append,
            )

        slot = self._acquire_fresh_slot()
        return PrefillPlan(
            tail_tokens=list(request_tokens),
            reset=True,
            reused_prefix=0,
            slot=slot,
        )

    def commit_prompt(self, plan: PrefillPlan, request_tokens: list[int]) -> None:
        """Record the full prompt as the chosen slot's contents."""
        plan.slot.tokens = list(request_tokens)

    def commit_generated(
        self, plan: PrefillPlan, generated_tokens: list[int]
    ) -> None:
        """Append freshly sampled tokens to the slot's token history."""
        plan.slot.tokens.extend(generated_tokens)

    def best_partial_match(
        self, request_tokens: list[int]
    ) -> tuple[CacheSlot | None, int]:
        """Return the slot sharing the longest prefix with ``request_tokens``.

        Unlike :meth:`plan_request`, this does not require the slot's
        full token history to be a prefix of the request — it reports
        the point where any slot's stored tokens diverge from the
        incoming ones. Used purely for diagnostics: if this returns a
        long partial match while ``plan_request`` returns ``reset=True``,
        the client is mutating a prefix that could otherwise be reused.
        """
        best: CacheSlot | None = None
        best_len = 0
        for slot in self.slots:
            if not slot.tokens:
                continue
            prefix = _common_prefix_len(slot.tokens, request_tokens)
            if prefix > best_len:
                best = slot
                best_len = prefix
        return best, best_len

    def reset(self) -> None:
        """Drop every slot."""
        self.slots = []
        self._counter = 0
        self._next_slot_id = 0

    def _acquire_fresh_slot(self) -> CacheSlot:
        """Either allocate a new slot or evict LRU and reuse it."""
        if len(self.slots) < self.max_slots:
            slot = CacheSlot(
                cache=self.middleware.create_cache(),
                tokens=[],
                last_used=self._counter,
                slot_id=self._next_slot_id,
            )
            self._next_slot_id += 1
            self.slots.append(slot)
            logger.debug(
                "allocated new slot %d (pool size %d/%d)",
                slot.slot_id, len(self.slots), self.max_slots,
            )
            return slot

        victim = min(self.slots, key=lambda s: s.last_used)
        logger.debug(
            "evicting LRU slot %d (last_used=%d, tokens=%d)",
            victim.slot_id, victim.last_used, len(victim.tokens),
        )
        victim.cache = self.middleware.create_cache()
        victim.tokens = []
        victim.last_used = self._counter
        victim.slot_id = self._next_slot_id
        self._next_slot_id += 1
        return victim


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Return the length of the shared prefix of two token lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
