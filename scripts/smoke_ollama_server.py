"""End-to-end smoke test for the KIV ollama-compatible server.

Loads Gemma 4 E2B IT in-process, starts the FastAPI server on a local
port, drives it through the ollama protocol with httpx, and validates:

  * /api/version and /api/tags respond
  * /api/chat streams NDJSON with a final ``done: true`` chunk
  * A second /api/chat turn reuses the prior KV cache (only the new
    tail is prefilled, not the full history)
  * /api/generate non-streaming returns a single JSON body
  * Basic timing information is reported

Run:

    python scripts/smoke_ollama_server.py --quantize 4bit

Skip load entirely (e.g. on CI without GPU) by pointing at a dummy
model with --dry-run, which just exercises the HTTP stubs used in
tests/test_server_app.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435  # intentionally not 11434 so a real ollama can coexist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--quantize", default="4bit", choices=("4bit", "8bit", "none"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--hot-budget", type=int, default=2048)
    parser.add_argument("--top-p-kiv", type=int, default=256)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the stub model from tests (no GPU, no download).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("smoke")

    loaded = _load_model(args, log)
    server = _start_server(loaded, args.host, args.port, log)

    try:
        _run_checks(args, log)
    finally:
        log.info("shutting server down")
        server.should_exit = True
        server.thread.join(timeout=10)

    log.info("smoke test PASSED")
    return 0


def _load_model(args: argparse.Namespace, log: logging.Logger) -> Any:
    if args.dry_run:
        log.info("dry run: using stub model (no GPU, no download)")
        from tests.test_server_app import (  # type: ignore[import-not-found]
            _StubLoadedModel,
            _StubMiddleware,
            _StubModel,
            _StubTokenizer,
        )
        stub_model = _StubModel()
        stub_middleware = _StubMiddleware()
        # The reuse-path prefill calls ``middleware.model``; wire it here
        # so any dry-run exercise of that path behaves like the test
        # fixture rather than raising a NoneType call.
        stub_middleware.model = stub_model
        return _StubLoadedModel(
            model=stub_model,
            tokenizer=_StubTokenizer(),
            middleware=stub_middleware,
        )

    from kiv.config import KIVConfig
    from kiv.server.model_loader import load_model

    quant = None if args.quantize == "none" else args.quantize
    log.info("loading %s (quantize=%s, dtype=%s)", args.model, quant, args.dtype)
    t0 = time.perf_counter()
    loaded = load_model(
        args.model,
        quantize=quant,
        dtype=args.dtype,
        kiv_config=KIVConfig(
            hot_budget=args.hot_budget, top_p=args.top_p_kiv
        ),
    )
    log.info("model loaded in %.1fs", time.perf_counter() - t0)
    return loaded


def _start_server(
    loaded: Any, host: str, port: int, log: logging.Logger
) -> Any:
    """Start uvicorn in a thread; return the uvicorn.Server (with .thread)."""
    import uvicorn

    from kiv.server.app import create_app

    app = create_app(loaded)
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", loop="asyncio"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    thread.start()
    server.thread = thread  # type: ignore[attr-defined]

    _wait_ready(host, port, log)
    return server


def _wait_ready(host: str, port: int, log: logging.Logger, timeout: float = 30.0) -> None:
    import httpx

    url = f"http://{host}:{port}/api/version"
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                log.info("server ready at %s", url)
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server did not become ready within {timeout}s")


def _run_checks(args: argparse.Namespace, log: logging.Logger) -> None:
    import httpx

    base = f"http://{args.host}:{args.port}"
    client = httpx.Client(base_url=base, timeout=300.0)

    log.info("GET /api/version")
    r = client.get("/api/version")
    r.raise_for_status()
    log.info("  -> %s", r.json())

    log.info("GET /api/tags")
    r = client.get("/api/tags")
    r.raise_for_status()
    tags = r.json()
    log.info("  -> %d model(s): %s", len(tags["models"]), tags["models"][0]["name"])

    log.info("POST /api/chat (turn 1, streaming)")
    turn1 = _chat_streaming(
        client,
        model=tags["models"][0]["name"],
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        num_predict=args.num_predict,
    )
    log.info(
        "  -> %d tokens, prompt_eval=%d (%.1fms), eval=%d (%.1fms)",
        turn1["eval_count"],
        turn1["prompt_eval_count"],
        turn1["prompt_eval_duration"] / 1e6,
        turn1["eval_count"],
        turn1["eval_duration"] / 1e6,
    )
    log.info("  assistant: %s", _truncate(turn1["text"]))

    log.info("POST /api/chat (turn 2, expect prefix reuse)")
    turn2 = _chat_streaming(
        client,
        model=tags["models"][0]["name"],
        messages=[
            {"role": "user", "content": "Say hello in one short sentence."},
            {"role": "assistant", "content": turn1["text"]},
            {"role": "user", "content": "Now say goodbye."},
        ],
        num_predict=args.num_predict,
    )
    log.info(
        "  -> %d tokens, prompt_eval=%d (%.1fms), eval=%d (%.1fms)",
        turn2["eval_count"],
        turn2["prompt_eval_count"],
        turn2["prompt_eval_duration"] / 1e6,
        turn2["eval_count"],
        turn2["eval_duration"] / 1e6,
    )
    log.info("  assistant: %s", _truncate(turn2["text"]))

    # Prefix reuse check. Real HuggingFace chat templates produce
    # append-friendly token sequences: turn N's tokens strictly extend
    # turn N-1's tokens plus the assistant's closing EOS, so reuse is a
    # hard invariant. The dry-run stub template does not have that
    # property, so the assertion only runs outside dry-run mode.
    turn1_prompt_tokens = turn1["prompt_eval_count"]
    bound = turn1_prompt_tokens + turn1["eval_count"]
    reused = turn2["prompt_eval_count"] < bound
    if args.dry_run:
        log.info(
            "  dry run: prefix reuse = %s (turn2 prefill=%d, bound=%d) — "
            "stub template does not compose like real chat templates, "
            "so failure here is expected",
            reused, turn2["prompt_eval_count"], bound,
        )
    else:
        if not reused:
            raise AssertionError(
                f"expected prefix reuse: turn2 prefill={turn2['prompt_eval_count']} "
                f">= full-replay bound {bound}"
            )
        log.info(
            "  prefix reuse confirmed: turn2 prefill=%d < full-replay bound=%d",
            turn2["prompt_eval_count"],
            bound,
        )

    log.info("POST /api/generate (non-streaming)")
    r = client.post(
        "/api/generate",
        json={
            "model": tags["models"][0]["name"],
            "prompt": "The capital of France is",
            "stream": False,
            "options": {"num_predict": 8, "temperature": 0.0},
        },
    )
    r.raise_for_status()
    body = r.json()
    assert body["done"] is True, f"expected done=true, got {body}"
    log.info("  -> response: %r", _truncate(body["response"]))


def _chat_streaming(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    num_predict: int,
) -> dict[str, Any]:
    """Send a streaming /api/chat request; return collected text + timings."""
    with client.stream(
        "POST",
        "/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": num_predict, "temperature": 0.0},
        },
    ) as response:
        response.raise_for_status()
        parts: list[str] = []
        final: dict[str, Any] = {}
        for line in response.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("done"):
                final = obj
                break
            parts.append(obj["message"]["content"])

    if not final:
        raise RuntimeError("stream ended without done=true chunk")

    return {
        "text": "".join(parts),
        "prompt_eval_count": final.get("prompt_eval_count", 0),
        "prompt_eval_duration": final.get("prompt_eval_duration", 0),
        "eval_count": final.get("eval_count", 0),
        "eval_duration": final.get("eval_duration", 0),
    }


def _truncate(text: str, n: int = 200) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


if __name__ == "__main__":
    sys.exit(main())
