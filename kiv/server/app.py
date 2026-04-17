"""FastAPI app exposing an ollama-compatible API backed by KIV.

Endpoints implemented (subset, matches what common ollama clients use):
  * POST /api/chat      - streaming or buffered chat completion
  * POST /api/generate  - streaming or buffered raw completion
  * GET  /api/tags      - list the one loaded model
  * POST /api/show      - model metadata
  * GET  /api/version   - ollama-compatible version stub

Concurrency: a single asyncio lock serializes all generation calls. The
server is designed for single-user consumer hardware (the same context
where a 4070 cannot usefully serve parallel generations).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .generation import SamplingParams, generate_stream, prefill
from .schemas import ChatRequest, GenerateRequest, ShowRequest
from .session import KIVSession

logger = logging.getLogger(__name__)

_KIV_SERVER_VERSION = "kiv-0.1.0"
_OLLAMA_COMPAT_VERSION = "0.1.48"
_SENTINEL_DONE = object()
_SENTINEL_ERROR = object()


def create_app(
    loaded_model: Any,
    *,
    max_slots: int = 8,
    prefill_chunk_size: int = 4096,
    prefill_hot_cap: int | None = 4096,
    debug_request_dir: str | None = None,
    debug_head_tokens: int = 0,
) -> FastAPI:
    """Build a FastAPI app bound to a single preloaded KIV model.

    ``loaded_model`` is a :class:`kiv.server.model_loader.LoadedModel`.
    The parameter is typed as ``Any`` so importers do not transitively
    depend on ``transformers``.

    ``max_slots`` sizes the multi-cache pool. A chat UI frequently
    issues auxiliary ``/api/chat`` calls (title generation, tag
    generation, follow-up suggestions) whose prompts bear no relation
    to the main conversation. The pool keeps each prefix in its own
    slot so background calls cannot evict the user's warm chat.

    ``prefill_hot_cap`` caps hot-cache size during fresh-cache prefill.
    Without the cap, hot grows with every chunk and per-chunk attention
    is O(chunk x total_so_far), making bulk prefill quadratic. Passing
    ``None`` disables the cap; prefill then retains full attention across
    the prompt at the cost of quadratic runtime.

    ``debug_request_dir`` enables full request dumps: on every /api/chat
    and /api/generate call, the decoded prompt is written to a
    timestamped file under the given directory. Intended for diagnosing
    prefix-drift cases where a client mutates the start of its prompts
    and defeats slot reuse.

    ``debug_head_tokens`` logs the first N decoded tokens of every
    request inline. Set to 0 to disable. Useful for spotting dynamic
    preambles (timestamps, session ids) in the server log without
    writing anything to disk.
    """
    app = FastAPI(title="KIV", version=_KIV_SERVER_VERSION)

    state = _ServerState(
        loaded_model,
        max_slots=max_slots,
        prefill_chunk_size=prefill_chunk_size,
        prefill_hot_cap=prefill_hot_cap,
        debug_request_dir=debug_request_dir,
        debug_head_tokens=debug_head_tokens,
    )
    app.state.kiv_state = state

    _register_routes(app)
    return app


class _ServerState:
    """Mutable state that endpoints share."""

    def __init__(
        self,
        loaded_model: Any,
        *,
        max_slots: int = 8,
        prefill_chunk_size: int = 4096,
        prefill_hot_cap: int | None = 4096,
        debug_request_dir: str | None = None,
        debug_head_tokens: int = 0,
    ) -> None:
        self.loaded_model = loaded_model
        self.session = KIVSession(
            middleware=loaded_model.middleware, max_slots=max_slots
        )
        self.generation_lock = asyncio.Lock()
        self.device = next(loaded_model.model.parameters()).device
        self.eos_ids = _collect_eos_ids(loaded_model.tokenizer, loaded_model.model)
        self.loaded_at = datetime.now(timezone.utc)
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_hot_cap = prefill_hot_cap
        self.debug_request_dir = debug_request_dir
        self.debug_head_tokens = debug_head_tokens
        self._debug_counter = 0
        if debug_request_dir is not None:
            import os
            os.makedirs(debug_request_dir, exist_ok=True)


def _register_routes(app: FastAPI) -> None:
    @app.get("/api/version")
    async def get_version() -> dict[str, str]:
        return {"version": _OLLAMA_COMPAT_VERSION}

    @app.get("/api/tags")
    async def get_tags(request: Request) -> dict[str, Any]:
        state = _state(request)
        info = _model_info(state)
        return {"models": [info]}

    @app.get("/api/kiv/stats")
    async def get_kiv_stats(request: Request) -> dict[str, Any]:
        """Retrieval telemetry across every pooled slot.

        For each slot's cold stores, returns aggregate score
        concentrations, average cold length, and the most recent
        retrieval calls with the page and token indices that were
        pulled back. Joining ``selected_token_indices`` against a known
        needle position yields an exact retrieval hit rate.

        This is a KIV-specific diagnostic endpoint, not part of the
        ollama API surface.
        """
        state = _state(request)
        slots_report = []
        for slot in state.session.slots:
            cold_stores = getattr(slot.cache, "cold_stores", {}) or {}
            per_layer = {
                layer_idx: cs.telemetry_snapshot()
                for layer_idx, cs in cold_stores.items()
            }
            slots_report.append({
                "slot_id": slot.slot_id,
                "tokens": len(slot.tokens),
                "last_used": slot.last_used,
                "cold_stores": per_layer,
            })
        return {
            "pool_size": len(state.session.slots),
            "max_slots": state.session.max_slots,
            "slots": slots_report,
        }

    @app.post("/api/kiv/stats/reset")
    async def reset_kiv_stats(request: Request) -> dict[str, Any]:
        """Clear telemetry ring buffers (and aggregates) on every slot.

        Call this before running a targeted retrieval measurement so
        previous traffic does not skew the hit-rate numbers.
        """
        state = _state(request)
        cleared = 0
        for slot in state.session.slots:
            for cs in getattr(slot.cache, "cold_stores", {}).values():
                cs.reset_telemetry()
                cleared += 1
        return {"cleared_cold_stores": cleared}

    @app.get("/api/ps")
    async def get_ps(request: Request) -> dict[str, Any]:
        """Ollama's 'currently loaded models' endpoint.

        KIV holds exactly one model, loaded at startup, so the response
        is the single loaded model with ``expires_at`` pushed far into
        the future and ``size_vram`` set to its parameter count.
        """
        state = _state(request)
        info = _model_info(state)
        info["expires_at"] = "9999-12-31T23:59:59Z"
        info["size_vram"] = info["size"]
        return {"models": [info]}

    @app.post("/api/show")
    async def post_show(payload: ShowRequest, request: Request) -> dict[str, Any]:
        state = _state(request)
        requested = payload.model or payload.name
        if requested and requested.split(":")[0] != state.loaded_model.display_name.split(":")[0]:
            raise HTTPException(
                status_code=404,
                detail=f"model '{requested}' not loaded; server holds "
                f"'{state.loaded_model.display_name}'",
            )
        info = _model_info(state)
        family = info["details"]["family"]
        # The ollama ``model_info`` section flattens keys under the model
        # family (for example ``llama.context_length``). Clients inspect
        # ``{family}.context_length`` to pick a default num_ctx.
        model_info_flat = {
            "general.architecture": family,
            "general.parameter_count": info["size"],
            f"{family}.context_length": info["context_length"],
        }
        return {
            "modelfile": f"# KIV: {state.loaded_model.repo_id}\n",
            "parameters": f"num_ctx {info['context_length']}\n",
            "template": _chat_template_or_empty(state.loaded_model.tokenizer),
            "details": info["details"],
            "model_info": model_info_flat,
            "capabilities": ["completion"],
        }

    @app.post("/api/chat")
    async def post_chat(payload: ChatRequest, request: Request):
        state = _state(request)
        return await _handle_generation(
            state=state,
            payload=payload,
            mode="chat",
        )

    @app.post("/api/generate")
    async def post_generate(payload: GenerateRequest, request: Request):
        state = _state(request)
        return await _handle_generation(
            state=state,
            payload=payload,
            mode="generate",
        )


def _state(request: Request) -> _ServerState:
    return request.app.state.kiv_state


async def _handle_generation(
    state: _ServerState, payload: Any, mode: str
) -> Any:
    """Unified entry point for /api/chat and /api/generate."""
    tokenizer = state.loaded_model.tokenizer
    input_tokens = _tokenize(payload, tokenizer, mode=mode)

    if not input_tokens:
        raise HTTPException(status_code=400, detail="empty prompt")

    params = SamplingParams.from_options(payload.options)
    stream_flag = getattr(payload, "stream", True)

    if stream_flag:
        generator = _stream_response(
            state, input_tokens, params, mode=mode,
        )
        return StreamingResponse(generator, media_type="application/x-ndjson")

    return await _buffered_response(state, input_tokens, params, mode=mode)


def _tokenize(payload: Any, tokenizer: Any, mode: str) -> list[int]:
    """Return token ids for a chat- or generate-shaped request."""
    if mode == "chat":
        messages = [m.model_dump() for m in payload.messages]
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        return _coerce_to_int_list(tokens)

    if getattr(payload, "raw", False):
        text = payload.prompt
    else:
        system = payload.system or ""
        text = f"{system}\n\n{payload.prompt}" if system else payload.prompt
    encoded = tokenizer(text, return_tensors=None, add_special_tokens=True)
    return _coerce_to_int_list(encoded["input_ids"])


def _coerce_to_int_list(tokens: Any) -> list[int]:
    """Normalize tokenizer output to a flat list of ints.

    Tokenizer outputs vary across versions and call sites: bare lists,
    nested lists for batches, dicts, ``BatchEncoding`` or
    ``BatchFeature`` objects, torch tensors. This helper flattens all
    shapes to a single list of integer token ids.

    A dict-like container without an ``input_ids`` key is a programming
    error upstream (we cannot recover a token stream from ``pixel_values``
    alone, for instance); in that case we raise a typed error rather
    than silently iterating over dict keys and failing with an opaque
    ``invalid literal for int()``.
    """
    if tokens is None:
        return []
    if hasattr(tokens, "keys") and not hasattr(tokens, "ndim"):
        try:
            has_input_ids = "input_ids" in tokens
        except TypeError:
            has_input_ids = False
        if has_input_ids:
            return _coerce_to_int_list(tokens["input_ids"])
        try:
            available = list(tokens.keys())
        except Exception:  # pragma: no cover - defensive
            available = []
        raise TypeError(
            "tokenizer returned a dict-like object without an 'input_ids' "
            f"field; available keys: {available}"
        )
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(t) for t in tokens]


async def _stream_response(
    state: _ServerState,
    input_tokens: list[int],
    params: SamplingParams,
    *,
    mode: str,
) -> AsyncIterator[bytes]:
    """NDJSON token-at-a-time streamer.

    The generation worker runs on a background thread. The streamer
    guarantees three things on exit (normal completion, client
    disconnect, or generation error):

    * the worker thread has stopped touching ``plan.slot.cache``
      before the async lock is released, so the next request cannot
      race with an orphaned forward pass;
    * the slot's token history is only committed on clean completion;
    * on abnormal exit the slot is discarded so a partial cache can
      never be reused as a prefix match by a later request.
    """
    async with state.generation_lock:
        q: queue.Queue = queue.Queue()
        cancel_event = threading.Event()
        start = time.perf_counter_ns()
        load_ns = 0

        plan = state.session.plan_request(input_tokens)
        logger.info(
            "request mode=%s tail=%d reused=%d reset=%s slot=%d pool=%d",
            mode,
            len(plan.tail_tokens),
            plan.reused_prefix,
            plan.reset,
            plan.slot.slot_id,
            len(state.session.slots),
        )
        _debug_log_request(state, mode, input_tokens, plan)

        # Point the KIV attention wrapper at this slot's cache before
        # any forward passes run. Other slots' caches remain allocated
        # but inactive until they are chosen by a later request.
        state.loaded_model.middleware.activate_cache(plan.slot.cache)

        stop_reason_box: list[str] = []
        thread = threading.Thread(
            target=_generation_worker,
            args=(state, plan, params, q, cancel_event, stop_reason_box),
            daemon=True,
        )
        thread.start()

        tokenizer = state.loaded_model.tokenizer
        model_name = state.loaded_model.display_name
        generated_ids: list[int] = []
        prompt_eval_ns = 0
        eval_start_ns: int | None = None
        clean_exit = False
        last_err: Exception | None = None

        try:
            while True:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, q.get
                )

                if isinstance(item, dict) and item.get("_event") == "prefill_done":
                    prompt_eval_ns = item["duration_ns"]
                    eval_start_ns = time.perf_counter_ns()
                    continue

                if item is _SENTINEL_ERROR:
                    last_err = await asyncio.get_event_loop().run_in_executor(
                        None, q.get
                    )
                    yield _ndjson(
                        {
                            "model": model_name,
                            "created_at": _now_iso(),
                            "error": str(last_err),
                            "done": True,
                        }
                    )
                    return

                if item is _SENTINEL_DONE:
                    clean_exit = True
                    break

                token_id: int = item  # type: ignore[assignment]
                generated_ids.append(token_id)
                text = tokenizer.decode([token_id], skip_special_tokens=True)

                chunk = {
                    "model": model_name,
                    "created_at": _now_iso(),
                    "done": False,
                }
                if mode == "chat":
                    chunk["message"] = {"role": "assistant", "content": text}
                else:
                    chunk["response"] = text
                yield _ndjson(chunk)

            # Commit is only safe after a clean SENTINEL_DONE because
            # the KV cache and the slot's token history must agree.
            state.session.commit_prompt(plan, input_tokens)
            state.session.commit_generated(plan, generated_ids)

            total_ns = time.perf_counter_ns() - start
            eval_ns = (
                time.perf_counter_ns() - eval_start_ns
                if eval_start_ns is not None
                else 0
            )
            done_reason = stop_reason_box[0] if stop_reason_box else "stop"
            final = {
                "model": model_name,
                "created_at": _now_iso(),
                "done": True,
                "total_duration": total_ns,
                "load_duration": load_ns,
                "prompt_eval_count": len(plan.tail_tokens),
                "prompt_eval_duration": prompt_eval_ns,
                "eval_count": len(generated_ids),
                "eval_duration": eval_ns,
                "done_reason": done_reason,
            }
            if mode == "chat":
                final["message"] = {"role": "assistant", "content": ""}
            else:
                final["response"] = ""
                final["context"] = list(plan.slot.tokens)
            yield _ndjson(final)
        finally:
            cancel_event.set()
            # Drain any queued items so the worker cannot block on
            # ``q.put`` while we wait for it to observe the cancel
            # signal. ``queue.Queue`` is unbounded so put() never
            # actually blocks, but draining is cheap insurance.
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            # Wait for the worker via the thread pool so neither the
            # event loop nor other endpoints stall while this request's
            # worker finishes its current forward pass. Prefill is not
            # interruptible, so a cancelled long prefill will block
            # THIS request's cleanup, but the rest of the server stays
            # responsive.
            await asyncio.get_event_loop().run_in_executor(None, thread.join)
            if not clean_exit:
                # The slot's KV cache may have been partially mutated
                # by the worker before the cancel landed. Drop the
                # slot so its tokens can never be prefix-matched by a
                # later request.
                _discard_slot(state, plan.slot)


async def _buffered_response(
    state: _ServerState,
    input_tokens: list[int],
    params: SamplingParams,
    *,
    mode: str,
) -> JSONResponse:
    """Non-streaming path. Collects tokens then returns once."""
    chunks: list[bytes] = []
    async for chunk in _stream_response(state, input_tokens, params, mode=mode):
        chunks.append(chunk)

    # Reassemble text.
    text_parts: list[str] = []
    final_obj: dict[str, Any] = {}
    for raw in chunks:
        obj = json.loads(raw.decode("utf-8"))
        if obj.get("done"):
            final_obj = obj
            continue
        if mode == "chat":
            text_parts.append(obj["message"]["content"])
        else:
            text_parts.append(obj["response"])

    full_text = "".join(text_parts)
    if mode == "chat":
        final_obj["message"] = {"role": "assistant", "content": full_text}
    else:
        final_obj["response"] = full_text

    return JSONResponse(final_obj)


def _generation_worker(
    state: _ServerState,
    plan: Any,
    params: SamplingParams,
    q: "queue.Queue[Any]",
    cancel_event: threading.Event,
    stop_reason_box: list[str],
) -> None:
    """Thread target: runs prefill + decode, pushes tokens onto the queue.

    ``cancel_event`` is polled between decode steps. When set, the
    worker stops issuing further forward passes and exits. Prefill
    itself is not interruptible; long prefills must run to completion
    before cancellation takes effect.

    ``stop_reason_box`` is a single-element mutable sink through which
    :func:`generate_stream` reports why it ended (eos, stop, length,
    cancel). The streamer reads it to set ollama's ``done_reason``.
    """
    try:
        prefill_start = time.perf_counter_ns()
        last_logits = prefill(
            state.loaded_model.middleware,
            plan.slot.cache,
            plan.tail_tokens,
            device=state.device,
            fresh_cache=plan.reset,
            chunk_size=state.prefill_chunk_size,
            prefill_hot_cap=state.prefill_hot_cap,
        )
        prefill_ns = time.perf_counter_ns() - prefill_start
        q.put({"_event": "prefill_done", "duration_ns": prefill_ns})

        if cancel_event.is_set():
            stop_reason_box.append("cancel")
            return

        token_iter = generate_stream(
            state.loaded_model.model,
            plan.slot.cache,
            last_logits,
            params=params,
            eos_ids=state.eos_ids,
            device=state.device,
            cancel_event=cancel_event,
            tokenizer=state.loaded_model.tokenizer,
            stop_reason_out=stop_reason_box,
        )
        for token_id in token_iter:
            if cancel_event.is_set():
                return
            q.put(token_id)
        q.put(_SENTINEL_DONE)
    except Exception as exc:  # noqa: BLE001
        logger.exception("generation failed")
        q.put(_SENTINEL_ERROR)
        q.put(exc)


def _debug_log_request(
    state: _ServerState,
    mode: str,
    input_tokens: list[int],
    plan: Any,
) -> None:
    """Emit diagnostics that expose why prefix reuse fires or misses.

    Two outputs, independently toggleable:

    * ``debug_head_tokens`` logs the first N decoded tokens of the
      request inline at INFO level. Dynamic preambles
      (timestamps, session ids, per-request headers) become visible
      immediately when you compare consecutive log lines.
    * ``debug_request_dir`` dumps the full decoded prompt to
      ``<dir>/req-<counter>-<mode>.txt`` for offline diffing when the
      head alone isn't enough to spot the drift.

    On a reset, the longest partial match against any existing slot is
    reported so the user can tell "entirely new conversation" apart
    from "almost the same as a prior turn, drifted at token N".
    """
    if plan.reset and state.session.slots:
        _, partial_len = state.session.best_partial_match(input_tokens)
        if partial_len > 0:
            logger.info(
                "reset with partial match: first %d/%d tokens matched "
                "an existing slot; drift starts at token %d",
                partial_len, len(input_tokens), partial_len,
            )

    if state.debug_head_tokens > 0:
        head = input_tokens[: state.debug_head_tokens]
        try:
            decoded = state.loaded_model.tokenizer.decode(
                head, skip_special_tokens=False
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic path
            decoded = f"<decode failed: {exc}>"
        logger.info("request head[%d]: %r", len(head), decoded)

    if state.debug_request_dir is not None:
        state._debug_counter += 1
        path = _debug_dump_path(state, mode)
        try:
            decoded_full = state.loaded_model.tokenizer.decode(
                input_tokens, skip_special_tokens=False
            )
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    f"# mode={mode} tokens={len(input_tokens)} "
                    f"reset={plan.reset} reused={plan.reused_prefix} "
                    f"slot={plan.slot.slot_id}\n"
                )
                fh.write(decoded_full)
            logger.info("wrote request dump to %s", path)
        except Exception as exc:  # noqa: BLE001 - diagnostic path
            logger.warning("request dump to %s failed: %s", path, exc)


def _debug_dump_path(state: _ServerState, mode: str) -> str:
    import os
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"req-{state._debug_counter:06d}-{stamp}-{mode}.txt"
    return os.path.join(state.debug_request_dir, filename)


def _discard_slot(state: _ServerState, slot: Any) -> None:
    """Remove a slot from the session pool.

    Called when a request exits without a clean completion (cancellation
    or worker error). The slot's KV cache may have been partially
    mutated, so its token history can no longer be trusted as a
    prefix-match target for future requests.
    """
    try:
        state.session.slots.remove(slot)
    except ValueError:
        pass
    state.loaded_model.middleware.activate_cache(None)


def _collect_eos_ids(tokenizer: Any, model: Any) -> set[int]:
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(x) for x in eos if x is not None)

    model_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(model_eos, int):
        ids.add(model_eos)
    elif isinstance(model_eos, (list, tuple)):
        ids.update(int(x) for x in model_eos if x is not None)
    return ids


def _model_info(state: _ServerState) -> dict[str, Any]:
    model_config = state.loaded_model.model.config
    text_config = getattr(model_config, "text_config", model_config)
    family = getattr(model_config, "model_type", "unknown")
    param_count = _count_params(state.loaded_model.model)
    native_ctx = _infer_context_length(text_config)
    return {
        "name": state.loaded_model.display_name,
        "model": state.loaded_model.display_name,
        "modified_at": state.loaded_at.isoformat(),
        "size": param_count,
        "digest": "kiv-live",
        "details": {
            "parent_model": state.loaded_model.repo_id,
            "format": "safetensors",
            "family": family,
            "families": [family],
            "parameter_size": f"{param_count / 1e9:.2f}B",
            "quantization_level": "none",
        },
        "context_length": native_ctx,
    }


def _infer_context_length(config: Any) -> int:
    """Best-effort max-position inference across HF config conventions."""
    for attr in ("max_position_embeddings", "max_seq_len", "n_positions"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 8192


def _count_params(model: Any) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:  # pragma: no cover - defensive
        return 0


def _chat_template_or_empty(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template
    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ndjson(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")
