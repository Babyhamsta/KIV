"""Regression tests for middleware prefill behavior."""

from types import SimpleNamespace

import torch

from kiv.middleware import KIVMiddleware, _prefill_logits_kwargs


class _FakeCache:
    def __init__(self):
        self._suppress_cold = False
        self._prefill_complete = False
        self.evict_calls = []
        self.marked = False

    def _evict_to_cap(self, cap):
        self.evict_calls.append(cap)

    def _evict_excess_all_layers(self):
        self.evicted_excess = True

    def mark_prefill_complete(self):
        self.marked = True


class _ModelWithLogitsToKeep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.calls = []

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
    ):
        self.calls.append(logits_to_keep)
        logits_len = logits_to_keep or input_ids.shape[1]
        logits = torch.zeros(input_ids.shape[0], logits_len, 3)
        return SimpleNamespace(logits=logits)


class _ModelWithoutLogitsToKeep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.calls = 0

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        self.calls += 1
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)
        return SimpleNamespace(logits=logits)


def _middleware_for_chunked_prefill(model):
    middleware = KIVMiddleware(model)
    middleware._installed = True
    middleware._original_impl = "eager"
    middleware._kiv_key = "kiv_eager"
    return middleware


def test_prefill_logits_kwargs_detects_supported_models():
    assert _prefill_logits_kwargs(_ModelWithLogitsToKeep()) == {"logits_to_keep": 1}


def test_prefill_logits_kwargs_skips_unsupported_models():
    assert _prefill_logits_kwargs(_ModelWithoutLogitsToKeep()) == {}


def test_chunked_prefill_uses_logits_to_keep_when_supported():
    model = _ModelWithLogitsToKeep()
    middleware = _middleware_for_chunked_prefill(model)
    cache = _FakeCache()

    logits = middleware.chunked_prefill(
        torch.ones(1, 5, dtype=torch.long),
        cache,
        chunk_size=2,
    )

    assert model.calls == [1, 1, 1]
    assert logits.shape == (1, 3)
    assert cache.marked is True


def test_chunked_prefill_falls_back_for_unsupported_models():
    model = _ModelWithoutLogitsToKeep()
    middleware = _middleware_for_chunked_prefill(model)
    cache = _FakeCache()

    logits = middleware.chunked_prefill(
        torch.ones(1, 5, dtype=torch.long),
        cache,
        chunk_size=2,
    )

    assert model.calls == 3
    assert logits.shape == (1, 3)
    assert cache.marked is True
