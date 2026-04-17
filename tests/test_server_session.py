"""Unit tests for KIVSession prefix-reuse and multi-slot pooling."""

from __future__ import annotations

import pytest

from kiv.server.session import KIVSession


class _FakeMiddleware:
    """Stand-in middleware. Each create_cache mints a unique sentinel."""

    def __init__(self) -> None:
        self.create_calls = 0
        self.activate_calls = 0

    def create_cache(self) -> object:
        self.create_calls += 1
        return ("cache", self.create_calls)

    def activate_cache(self, cache) -> None:
        self.activate_calls += 1


def test_first_request_allocates_first_slot() -> None:
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=2)

    plan = session.plan_request([1, 2, 3])

    assert plan.reset is True
    assert plan.reused_prefix == 0
    assert plan.tail_tokens == [1, 2, 3]
    assert len(session.slots) == 1
    assert mw.create_calls == 1


def test_pure_append_reuses_same_slot() -> None:
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=2)

    first = session.plan_request([1, 2, 3])
    session.commit_prompt(first, [1, 2, 3])
    session.commit_generated(first, [4, 5])

    second = session.plan_request([1, 2, 3, 4, 5, 6])

    assert second.reset is False
    assert second.reused_prefix == 5
    assert second.tail_tokens == [6]
    assert second.slot is first.slot
    assert mw.create_calls == 1


def test_aux_call_lands_in_second_slot_without_disturbing_first() -> None:
    """Simulates an Open WebUI title-generation call arriving between turns.

    The main chat cache must survive so the user's next real message
    still reuses its prefix.
    """
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=4)

    # Turn 1: main chat message.
    chat1 = session.plan_request([1, 2, 3, 4, 5])
    session.commit_prompt(chat1, [1, 2, 3, 4, 5])
    session.commit_generated(chat1, [6, 7])
    chat_slot = chat1.slot

    # Aux call: totally different tokens (title generation prompt).
    aux = session.plan_request([99, 98, 97])
    assert aux.slot is not chat_slot
    assert aux.reset is True
    session.commit_prompt(aux, [99, 98, 97])
    session.commit_generated(aux, [96])

    # Turn 2: user's next real message extending the original chat.
    chat2 = session.plan_request([1, 2, 3, 4, 5, 6, 7, 8])

    assert chat2.slot is chat_slot, "main chat cache was evicted by aux call"
    assert chat2.reset is False
    assert chat2.reused_prefix == 7
    assert chat2.tail_tokens == [8]


def test_lru_eviction_when_pool_is_full() -> None:
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=2)

    a = session.plan_request([1, 2])
    session.commit_prompt(a, [1, 2])

    b = session.plan_request([10, 20])
    session.commit_prompt(b, [10, 20])

    # Touch 'a' to make it the MRU.
    touch = session.plan_request([1, 2, 3])
    session.commit_prompt(touch, [1, 2, 3])
    assert touch.slot is a.slot

    # Third distinct prefix must evict LRU (which is now 'b').
    c = session.plan_request([100, 200])
    assert c.slot is b.slot
    assert c.reset is True


def test_divergence_on_same_slot_spawns_new_slot() -> None:
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=4)

    original = session.plan_request([1, 2, 3])
    session.commit_prompt(original, [1, 2, 3])
    session.commit_generated(original, [4])

    # User edits an earlier message -> divergence.
    edited = session.plan_request([1, 2, 9, 9])

    assert edited.reset is True
    assert edited.slot is not original.slot
    assert len(session.slots) == 2

    # Original slot still holds its tokens; future matches still work.
    resumed = session.plan_request([1, 2, 3, 4, 5])
    assert resumed.slot is original.slot
    assert resumed.reused_prefix == 4
    assert resumed.tail_tokens == [5]


def test_regen_request_truncates_slot_instead_of_re_prefilling() -> None:
    """Chat regeneration sends the same tokens *minus* the last assistant
    response. The slot still holds the full prior sequence, so the
    incoming request is a strict prefix of the slot. The session should
    roll the cache back in place and prefill only one token, not
    allocate a fresh slot and re-prefill the whole prompt.
    """

    class _TruncatingCache:
        def __init__(self) -> None:
            self.truncated_to: int | None = None

        def truncate_to(self, new_len: int) -> bool:
            self.truncated_to = new_len
            return True

    class _TruncatingMiddleware:
        def __init__(self) -> None:
            self.caches: list[_TruncatingCache] = []

        def create_cache(self) -> _TruncatingCache:
            cache = _TruncatingCache()
            self.caches.append(cache)
            return cache

    mw = _TruncatingMiddleware()
    session = KIVSession(middleware=mw, max_slots=4)

    first = session.plan_request([1, 2, 3, 4, 5])
    session.commit_prompt(first, [1, 2, 3, 4, 5])
    session.commit_generated(first, [6, 7, 8])
    # Slot now holds [1..5] prompt + [6..8] generated.

    regen = session.plan_request([1, 2, 3, 4, 5])

    assert regen.reset is False
    assert regen.slot is first.slot
    assert regen.tail_tokens == [5]
    assert regen.reused_prefix == 4
    assert first.slot.cache.truncated_to == 4
    assert len(session.slots) == 1


def test_truncate_match_skipped_when_cache_rejects_truncation() -> None:
    """If the underlying cache reports that truncation would require
    touching cold storage, the session falls back to allocating a
    fresh slot instead of corrupting the existing one.
    """

    class _RejectingCache:
        def truncate_to(self, new_len: int) -> bool:
            return False

    class _RejectingMiddleware:
        def create_cache(self):
            return _RejectingCache()

    mw = _RejectingMiddleware()
    session = KIVSession(middleware=mw, max_slots=4)

    first = session.plan_request([1, 2, 3, 4, 5])
    session.commit_prompt(first, [1, 2, 3, 4, 5])
    session.commit_generated(first, [6, 7, 8])

    regen = session.plan_request([1, 2, 3, 4, 5])

    assert regen.reset is True
    assert regen.slot is not first.slot
    assert len(session.slots) == 2


def test_exact_duplicate_request_triggers_fresh_slot() -> None:
    """An incoming request that exactly matches a slot's full token
    history cannot be served from that slot: the slot's KV cache has
    no position-level truncation primitive, so reusing it and
    "reprefilling" the last token would append a duplicate position
    and silently desynchronise ``slot.tokens`` from the cache. A
    fresh slot is allocated instead.
    """
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=4)

    first = session.plan_request([1, 2, 3])
    session.commit_prompt(first, [1, 2, 3])

    plan = session.plan_request([1, 2, 3])

    assert plan.reset is True
    assert plan.reused_prefix == 0
    assert plan.tail_tokens == [1, 2, 3]
    assert plan.slot is not first.slot


def test_empty_request_rejected() -> None:
    session = KIVSession(middleware=_FakeMiddleware())
    with pytest.raises(ValueError):
        session.plan_request([])


def test_shorter_request_than_any_slot_allocates_new_slot() -> None:
    """A request shorter than the stored tokens is a divergence."""
    mw = _FakeMiddleware()
    session = KIVSession(middleware=mw, max_slots=4)

    first = session.plan_request([1, 2, 3, 4])
    session.commit_prompt(first, [1, 2, 3, 4])

    plan = session.plan_request([1, 2])

    assert plan.reset is True
    assert plan.slot is not first.slot


def test_reset_clears_pool() -> None:
    session = KIVSession(middleware=_FakeMiddleware(), max_slots=2)
    plan = session.plan_request([1, 2, 3])
    session.commit_prompt(plan, [1, 2, 3])

    session.reset()

    assert session.slots == []
