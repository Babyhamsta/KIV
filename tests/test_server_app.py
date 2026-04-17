"""Integration tests for the KIV ollama-compatible FastAPI app.

The tests stub out HuggingFace (model, tokenizer) and the KIV middleware
so they run without transformers, a GPU, or network access. They verify
the HTTP contract (NDJSON stream shape, JSON endpoints, status codes)
rather than end-to-end correctness of KIV itself — that is covered by
the existing KIV unit/evaluation tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient  # noqa: E402

from kiv.server.app import create_app  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


_EOS_TOKEN_ID = 0
_VOCAB = 256


class _StubTokenizer:
    """Char-level tokenizer. Token id == ord(char) + 1, 0 reserved for EOS."""

    eos_token_id = _EOS_TOKEN_ID
    chat_template = "{messages}"

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_tensors: Any = None,
    ):
        text_parts = [f"{m['role']}:{m['content']}" for m in messages]
        text = "|".join(text_parts)
        if add_generation_prompt:
            text += "|A:"
        tokens = self._encode(text)
        if not tokenize:
            return text
        return tokens

    def __call__(
        self,
        text: str,
        return_tensors: Any = None,
        add_special_tokens: bool = True,
    ):
        return {"input_ids": self._encode(text)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for tok in ids:
            if skip_special_tokens and tok == self.eos_token_id:
                continue
            chars.append(chr(max(0, tok - 1)))
        return "".join(chars)

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(c) + 1 for c in text]


class _StubParam:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def numel(self) -> int:
        return 1


class _StubModel:
    """Generates tokens following a fixed cycle for reproducible tests."""

    # Prefill emits X; decode cycles Y, Z, EOS.
    PREFILL_TOKEN = ord("X") + 1
    DECODE_CYCLE = [ord("Y") + 1, ord("Z") + 1, _EOS_TOKEN_ID]

    def __init__(self) -> None:
        self.config = SimpleNamespace(model_type="stub")
        self.generation_config = SimpleNamespace(eos_token_id=_EOS_TOKEN_ID)
        self._step = 0

    def parameters(self):
        return iter([_StubParam()])

    def __call__(self, *, input_ids, past_key_values, use_cache):
        logits = torch.full((1, input_ids.shape[1], _VOCAB), -1e4)
        next_id = self.DECODE_CYCLE[self._step % len(self.DECODE_CYCLE)]
        self._step += 1
        logits[:, -1, next_id] = 10.0
        return SimpleNamespace(logits=logits)


class _StubCache:
    """Minimal ``TieredKVCache`` stand-in for the server tests.

    ``generation.prefill`` toggles ``_prefill_complete`` on reuse-path
    forwards and calls ``_evict_excess_all_layers`` between chunks, so
    those attributes must exist on any cache used in integration tests.
    """

    def __init__(self, cache_id: int) -> None:
        self.id = cache_id
        self._prefill_complete = True

    def _evict_excess_all_layers(self) -> None:
        return None


class _StubMiddleware:
    def __init__(self) -> None:
        self.prefill_calls = 0
        self.activate_calls = 0
        self._caches = 0
        self.model = None  # set post-construction so direct model forward works

    def create_cache(self) -> Any:
        self._caches += 1
        return _StubCache(self._caches)

    def activate_cache(self, cache) -> None:
        self.activate_calls += 1

    def chunked_prefill(self, input_ids, cache, chunk_size: int = 4096, **_: Any):
        """Return logits pointing at the stub model's designated prefill token.

        Accepts and ignores the same kwargs the real middleware takes
        (prefill_hot_cap, empty_cache_interval) so tests don't break
        whenever the server passes a new prefill tuning argument.
        """
        self.prefill_calls += 1
        logits = torch.full((1, _VOCAB), -1e4)
        logits[:, _StubModel.PREFILL_TOKEN] = 10.0
        return logits


@dataclass
class _StubLoadedModel:
    model: _StubModel
    tokenizer: _StubTokenizer
    middleware: _StubMiddleware
    display_name: str = "stub:latest"
    repo_id: str = "stub/model"


@pytest.fixture()
def client() -> TestClient:
    model = _StubModel()
    middleware = _StubMiddleware()
    middleware.model = model  # generation.prefill(fresh_cache=False) reads this
    loaded = _StubLoadedModel(
        model=model,
        tokenizer=_StubTokenizer(),
        middleware=middleware,
    )
    app = create_app(loaded)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_version_endpoint(client: TestClient) -> None:
    response = client.get("/api/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_tags_endpoint_lists_loaded_model(client: TestClient) -> None:
    response = client.get("/api/tags")
    assert response.status_code == 200
    body = response.json()
    assert len(body["models"]) == 1
    assert body["models"][0]["name"] == "stub:latest"


def test_show_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/api/show", json={"model": "not-loaded"})
    assert response.status_code == 404


def test_show_loaded_model_returns_details(client: TestClient) -> None:
    response = client.post("/api/show", json={"model": "stub:latest"})
    assert response.status_code == 200
    body = response.json()
    assert body["details"]["family"] == "stub"


def test_num_predict_zero_emits_no_tokens(client: TestClient) -> None:
    """``num_predict=0`` must produce an empty response with
    ``done_reason=length``, not a single stray token."""
    payload = {
        "model": "stub:latest",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"num_predict": 0, "temperature": 0.0},
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    lines = [line for line in response.text.splitlines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["done"] is True
    assert chunks[-1]["done_reason"] == "length"
    assert chunks[-1]["eval_count"] == 0


def test_num_predict_capped_reports_length_done_reason(client: TestClient) -> None:
    """When generation runs until ``num_predict`` without hitting EOS,
    the stream must report ``done_reason=length`` so the client can
    tell natural stops apart from length-cap hits."""
    payload = {
        "model": "stub:latest",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"num_predict": 1, "temperature": 0.0},
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    chunks = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]
    assert chunks[-1]["done_reason"] == "length"
    assert chunks[-1]["eval_count"] == 1


def test_chat_stream_yields_ndjson_with_final_done(client: TestClient) -> None:
    payload = {
        "model": "stub:latest",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"num_predict": 8, "temperature": 0.0},
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    lines = [line for line in response.text.splitlines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["done"] is True
    # Stub model emits EOS as the third decode token, so the stream
    # terminates naturally (not via the length cap).
    assert chunks[-1]["done_reason"] == "eos"
    assert chunks[-1]["eval_count"] > 0

    body_text = "".join(c["message"]["content"] for c in chunks if not c["done"])
    # CYCLE = X, Y, Z, EOS -> decoded text is "XYZ" before EOS terminates stream.
    assert body_text == "XYZ"


def test_generate_non_streaming_returns_single_json(client: TestClient) -> None:
    payload = {
        "model": "stub:latest",
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 8, "temperature": 0.0},
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["done"] is True
    assert body["response"] == "XYZ"
    assert "context" in body


def test_append_reuses_same_slot(client: TestClient) -> None:
    """A raw /api/generate where the second prompt strictly extends
    (prompt + generated output + new tail) must reuse the same slot.

    Simulates how real HF chat templates compose: turn N's tokens
    include turn N-1's prompt AND the assistant's generated response.
    The stub model produces 'XY' deterministically at temp=0, so the
    follow-up prompt 'helloXYmore' is a valid extension of the stored
    slot tokens.
    """
    state = client.app.state.kiv_state
    middleware = state.loaded_model.middleware

    client.post(
        "/api/generate",
        json={
            "model": "stub:latest",
            "prompt": "hello",
            "raw": True,
            "stream": False,
            "options": {"num_predict": 2, "temperature": 0.0},
        },
    )
    first_slot_count = len(state.session.slots)
    prefill_calls_after_first = middleware.prefill_calls

    client.post(
        "/api/generate",
        json={
            "model": "stub:latest",
            # Includes the generated 'XY' so the slot's stored tokens
            # are a strict prefix of this request's tokens.
            "prompt": "helloXYmore",
            "raw": True,
            "stream": False,
            "options": {"num_predict": 2, "temperature": 0.0},
        },
    )

    # Same slot reused, no new slot allocated.
    assert len(state.session.slots) == first_slot_count
    # Reuse path uses a direct model forward, not chunked_prefill, so
    # the middleware's chunked_prefill counter does NOT advance. That
    # is the signal the slot was warm.
    assert middleware.prefill_calls == prefill_calls_after_first
    # activate_cache called once per request to bind the slot's cache.
    assert middleware.activate_calls >= 2


def test_aux_call_between_turns_does_not_evict_main_chat(client: TestClient) -> None:
    """Simulates Open WebUI's title-generation pattern.

    1) main chat fires with a user message
    2) an auxiliary call comes in with an unrelated prompt
       (title generation, tag extraction, follow-up suggestions)
    3) main chat fires again, extending the original conversation

    Step 3 must land on the original chat's warm slot, not spawn a
    new one, otherwise KIV loses the 1M-token warm cache every time
    the UI runs a background helper prompt.

    Raw /api/generate is used so the composition property is
    deterministic under the test stub. Real chat templates satisfy
    the same property; the live smoke test exercises /api/chat.
    """
    state = client.app.state.kiv_state
    middleware = state.loaded_model.middleware

    # Turn 1 of the main chat. Use a prompt that produces a
    # well-defined token prefix the extension can be built on.
    client.post(
        "/api/generate",
        json={
            "model": "stub:latest",
            "prompt": "main-chat-prefix",
            "raw": True,
            "stream": False,
            "options": {"num_predict": 2, "temperature": 0.0},
        },
    )

    # Auxiliary title-generation style call. Different prefix -> different slot.
    client.post(
        "/api/generate",
        json={
            "model": "stub:latest",
            "prompt": "Generate a 3-5 word title",
            "raw": True,
            "stream": False,
            "options": {"num_predict": 2, "temperature": 0.0},
        },
    )
    prefill_calls_after_aux = middleware.prefill_calls

    # Turn 2 of the main chat. Strictly extends turn 1's prefix +
    # the 'XY' the stub generator produced at temp=0.
    response = client.post(
        "/api/generate",
        json={
            "model": "stub:latest",
            "prompt": "main-chat-prefixXYcontinuation",
            "raw": True,
            "stream": False,
            "options": {"num_predict": 2, "temperature": 0.0},
        },
    )
    assert response.status_code == 200

    # Reuse path uses direct model forward, not chunked_prefill, so
    # chunked_prefill counter stays flat. If the aux call had evicted
    # the main chat slot, the main slot would be cold and we'd see
    # another chunked_prefill instead.
    assert middleware.prefill_calls == prefill_calls_after_aux
    # Pool should now hold exactly two slots: main chat and aux.
    assert len(state.session.slots) == 2
