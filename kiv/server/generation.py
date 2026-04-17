"""Token-at-a-time generation loop for the ollama-compatible server.

Lives outside ``app.py`` so prefill and sampling can be unit-tested
without spinning up FastAPI. Cache and token history are owned by
:class:`kiv.server.session.KIVSession`; this module drives the forward
pass and the sampling loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 512
# Hard ceiling used when a client asks for unlimited generation
# (ollama's ``num_predict = -1``). Relies on EOS to terminate naturally
# within this budget; prevents a runaway loop from never returning.
_UNBOUNDED_MAX_NEW_TOKENS = 16384


@dataclass
class SamplingParams:
    """Subset of generation options used during decode."""

    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    stop_strings: tuple[str, ...] = ()
    seed: int | None = None

    @classmethod
    def from_options(cls, options: Any) -> "SamplingParams":
        """Build from an ollama ``options`` payload. ``None`` allowed."""
        if options is None:
            return cls()
        if hasattr(options, "model_dump"):
            data = options.model_dump(exclude_none=True)
        elif isinstance(options, dict):
            data = {k: v for k, v in options.items() if v is not None}
        else:
            data = {}

        max_new = data.get("num_predict")
        # Ollama contract:
        #   None         -> use the server default
        #   -1 (or < 0)  -> "until EOS or context full" - we honour this
        #                   with a hard safety ceiling to guarantee
        #                   the request eventually returns
        #   >= 0         -> exact cap
        if max_new is None:
            max_new = _DEFAULT_MAX_NEW_TOKENS
        elif max_new < 0:
            max_new = _UNBOUNDED_MAX_NEW_TOKENS

        stop = data.get("stop") or ()
        return cls(
            max_new_tokens=int(max_new),
            temperature=float(data.get("temperature", 0.8)),
            top_p=float(data.get("top_p", 0.95)),
            top_k=int(data.get("top_k", 40)),
            stop_strings=tuple(stop),
            seed=data.get("seed"),
        )


def prefill(
    middleware: Any,
    cache: Any,
    tail_tokens: list[int],
    *,
    device: torch.device,
    chunk_size: int = 4096,
    fresh_cache: bool = True,
    prefill_hot_cap: int | None = None,
) -> torch.Tensor:
    """Run prefill on ``tail_tokens``; return logits for the last position.

    Two code paths:

    * ``fresh_cache=True``: dispatches to :meth:`KIVMiddleware.chunked_prefill`,
      which temporarily swaps in the model's native attention. The KIV
      wrapper has nothing to do on an empty cold store, and bypassing
      it avoids its per-call overhead.
    * ``fresh_cache=False``: the cache already holds prior-turn state
      (potentially with a populated cold store). The tail is forwarded
      through the model with the KIV wrapper live, so each chunk's
      queries can retrieve from cold. Chunking here is required for
      correctness, not performance - a single forward on a long tail
      builds a quadratic ``chunk x kv_length`` attention matrix.

    ``prefill_hot_cap`` (fresh path only) caps the hot cache after each
    chunk by evicting overflow to cold. Without the cap, hot grows
    every chunk and per-chunk attention is O(chunk x total_so_far),
    making bulk prefill quadratic in the prompt length. With the cap
    it's linear.
    """
    if not tail_tokens:
        raise ValueError("tail_tokens must be non-empty")

    input_ids = torch.tensor([tail_tokens], dtype=torch.long, device=device)

    if fresh_cache:
        kwargs: dict[str, Any] = {"chunk_size": chunk_size}
        if prefill_hot_cap is not None:
            kwargs["prefill_hot_cap"] = prefill_hot_cap
        logits = middleware.chunked_prefill(input_ids, cache, **kwargs)
        return logits  # [1, vocab]

    return _chunked_reuse_prefill(
        middleware.model,
        input_ids,
        cache,
        chunk_size=chunk_size,
    )


def _chunked_reuse_prefill(
    model: Any,
    input_ids: torch.Tensor,
    cache: Any,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Forward the tail through the model in bounded-size chunks.

    The KIV attention hook stays active so cold retrieval runs on every
    chunk. Three invariants keep the forward correct and bounded:

    * Inside each chunk, eviction is disabled (``_prefill_complete=False``).
      If eviction ran mid-forward, ``cache.update()`` could remove hot
      tokens that the chunk's own later query positions still need to
      attend to.
    * Between chunks, eviction is restored and overflow is flushed to
      cold. Hot size stays within ``hot_budget + chunk_size`` regardless
      of the total tail length.
    * The cold store's per-step candidate cache is invalidated before
      each chunk runs. With ``_prefill_complete=False`` the cache's
      ``_decode_step`` counter does not advance inside chunks, so shared-
      layer candidate reuse would otherwise hand chunk N the candidates
      computed for chunk N-1's queries - a silent correctness bug that
      only manifests when between-chunk eviction does not fire (short
      tail, cold-heavy cache).
    """
    total = input_ids.shape[1]
    last_logits: torch.Tensor | None = None
    prior_flag = getattr(cache, "_prefill_complete", True)

    try:
        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk = input_ids[:, start:end]

                _invalidate_cold_candidate_caches(cache)
                cache._prefill_complete = False
                outputs = model(
                    input_ids=chunk,
                    past_key_values=cache,
                    use_cache=True,
                )
                if end == total:
                    last_logits = outputs.logits[:, -1, :].clone()
                del outputs

                # Restore eviction and flush overflow before the next chunk.
                cache._prefill_complete = True
                if hasattr(cache, "_evict_excess_all_layers"):
                    cache._evict_excess_all_layers()
    finally:
        cache._prefill_complete = prior_flag

    if last_logits is None:  # pragma: no cover - input_ids guaranteed non-empty
        raise RuntimeError("prefill produced no logits")
    return last_logits


def _invalidate_cold_candidate_caches(cache: Any) -> None:
    """Reset per-step candidate caches on each cold store of ``cache``.

    Called before each chunk of a reuse-path prefill. See the docstring
    of :func:`_chunked_reuse_prefill` for why a stepless forward needs
    this invalidation.
    """
    cold_stores = getattr(cache, "cold_stores", None)
    if not cold_stores:
        return
    for store in cold_stores.values():
        store._cached_candidate_k = None
        store._cached_candidate_v = None
        store._cached_candidate_idx = None
        store._cached_step = -1


def generate_stream(
    model: Any,
    cache: Any,
    last_logits: torch.Tensor,
    *,
    params: SamplingParams,
    eos_ids: set[int],
    device: torch.device,
    cancel_event: Any = None,
    tokenizer: Any = None,
    stop_reason_out: list[str] | None = None,
) -> Iterator[int]:
    """Autoregressively sample tokens, yielding each one as it's produced.

    ``last_logits`` is the logits tensor returned by :func:`prefill` and
    is consumed to pick the first output token.

    ``cancel_event`` is an optional ``threading.Event``-like object. If
    its ``is_set()`` returns True between decode steps, the generator
    returns early without running the next forward pass. Callers use
    this to stop generation cooperatively when the HTTP client
    disconnects mid-stream.

    ``tokenizer`` is required only when ``params.stop_strings`` is set;
    the loop decodes a sliding window of recent tokens each step to
    detect stop substrings that may span token boundaries.

    ``stop_reason_out``, if provided, is populated with a single string
    describing why generation ended: ``"eos"`` (the model emitted an
    end-of-sequence token), ``"stop"`` (a user-supplied stop string
    matched), ``"length"`` (``max_new_tokens`` reached), or ``"cancel"``
    (the request was cancelled). Callers use this to set ollama's
    ``done_reason`` field correctly. Length-limited stops are
    distinguishable from natural stops via this field.
    """
    if params.max_new_tokens <= 0:
        if stop_reason_out is not None:
            stop_reason_out.append("length")
        return

    generator = _build_generator(params.seed, device)

    token = _sample(last_logits, params, generator)
    first_id = int(token.item())
    yield first_id
    if first_id in eos_ids:
        if stop_reason_out is not None:
            stop_reason_out.append("eos")
        return

    produced = 1
    next_input = token.view(1, 1)
    stop_buffer: list[int] = [first_id] if params.stop_strings else []
    stop_window = _stop_window_size(params.stop_strings)

    with torch.no_grad():
        while produced < params.max_new_tokens:
            if cancel_event is not None and cancel_event.is_set():
                if stop_reason_out is not None:
                    stop_reason_out.append("cancel")
                return

            outputs = model(
                input_ids=next_input,
                past_key_values=cache,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            del outputs

            token = _sample(logits, params, generator)
            token_id = int(token.item())
            yield token_id
            produced += 1
            if token_id in eos_ids:
                if stop_reason_out is not None:
                    stop_reason_out.append("eos")
                return
            if params.stop_strings and tokenizer is not None:
                stop_buffer.append(token_id)
                if _stop_string_hit(
                    stop_buffer, params.stop_strings, tokenizer, stop_window
                ):
                    if stop_reason_out is not None:
                        stop_reason_out.append("stop")
                    return
            next_input = token.view(1, 1)

    if stop_reason_out is not None:
        stop_reason_out.append("length")


def _stop_window_size(stop_strings: tuple[str, ...]) -> int:
    """Size (in tokens) of the tail window to decode for stop-string checks.

    A stop string can span multiple tokens, and a single token can
    decode to several characters or be a single character in edge
    cases. Decoding a window large enough to cover the longest stop
    string plus a modest buffer covers both. Decoding the entire
    accumulated output every step would be O(N^2) across the whole
    generation.
    """
    if not stop_strings:
        return 0
    longest = max(len(s) for s in stop_strings)
    # Assume one char per token worst case, then add a buffer so the
    # window is guaranteed to include the stop string even when it
    # appears right at the end.
    return max(longest, 16) + 16


def _stop_string_hit(
    token_buffer: list[int],
    stop_strings: tuple[str, ...],
    tokenizer: Any,
    window: int,
) -> bool:
    """Return True if any stop string appears in the decoded tail window.

    Only the last ``window`` tokens are decoded, bounding the per-step
    cost to a constant and making the stop-string check linear in the
    number of generated tokens instead of quadratic.
    """
    tail = token_buffer[-window:] if window > 0 else token_buffer
    text = tokenizer.decode(tail, skip_special_tokens=True)
    return any(s and s in text for s in stop_strings)


def _build_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Temperature + top-k + top-p sampling. Returns [1] long tensor."""
    logits = logits.squeeze(0) if logits.dim() == 2 else logits

    if params.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / max(params.temperature, 1e-5)

    if params.top_k and params.top_k > 0:
        top_k = min(params.top_k, logits.shape[-1])
        kth = torch.topk(logits, top_k, dim=-1).values[-1]
        logits = torch.where(
            logits < kth, torch.full_like(logits, float("-inf")), logits
        )

    if params.top_p is not None and 0.0 < params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > params.top_p
        # Always keep the top token.
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return token
