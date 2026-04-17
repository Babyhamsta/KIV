"""Command-line entry point: ``kiv serve --model <repo>``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from ..config import KIVConfig

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """argparse type that rejects values less than 1."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {parsed}"
        )
    return parsed


def _non_negative_int(value: str) -> int:
    """argparse type that rejects negative values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {parsed}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiv",
        description="KIV: long-context LLM server (ollama-compatible API).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve", help="Start an ollama-compatible HTTP server."
    )
    serve.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id or local path (e.g. google/gemma-4-E2B-it)",
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    serve.add_argument(
        "--port", type=int, default=11434, help="Bind port (default: 11434)"
    )
    serve.add_argument(
        "--name",
        default=None,
        help="Name surfaced via /api/tags (defaults to the repo id)",
    )
    serve.add_argument(
        "--quantize",
        choices=("4bit", "8bit"),
        default=None,
        help="Quantize via bitsandbytes. Requires 'kiv[quantization]'.",
    )
    serve.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Computation dtype (default: auto)",
    )
    serve.add_argument(
        "--device-map",
        default="auto",
        help="HuggingFace device_map for from_pretrained (default: auto)",
    )
    serve.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained.",
    )
    serve.add_argument(
        "--attn-impl",
        default="eager",
        help="attn_implementation passed to from_pretrained (default: eager)",
    )
    serve.add_argument(
        "--hot-budget",
        type=_positive_int,
        default=2048,
        help="KIV hot_budget (default: 2048)",
    )
    serve.add_argument(
        "--top-p-kiv",
        type=_positive_int,
        default=256,
        help="KIV top_p: cold tokens retrieved per decode step (default: 256)",
    )
    serve.add_argument(
        "--page-size",
        type=_positive_int,
        default=128,
        help="KIV page_size (default: 128)",
    )
    serve.add_argument(
        "--top-pages",
        type=_positive_int,
        default=32,
        help="KIV top_pages (default: 32)",
    )
    serve.add_argument(
        "--max-slots",
        type=_positive_int,
        default=8,
        help=(
            "Size of the cache pool (default: 8). Needed so aux /api/chat "
            "calls from the UI (title-gen, tag-gen, follow-up suggestions) "
            "don't evict the main conversation's warm cache. Open WebUI "
            "commonly fires 3-4 aux calls per message, each inlining the "
            "full chat, so a pool of 4 fills up quickly."
        ),
    )
    serve.add_argument(
        "--prefill-chunk-size",
        type=_positive_int,
        default=4096,
        help="Prefill chunk size (default: 4096).",
    )
    serve.add_argument(
        "--prefill-hot-cap",
        type=_non_negative_int,
        default=4096,
        help=(
            "Hot-cache cap during fresh-cache bulk prefill (default: 4096). "
            "Bounds per-chunk attention memory and keeps prefill linear "
            "instead of quadratic. Pass 0 to disable bounded prefill."
        ),
    )
    serve.add_argument(
        "--debug-head-tokens",
        type=_non_negative_int,
        default=0,
        help=(
            "Log the first N decoded tokens of every incoming request "
            "at INFO level. Useful for spotting dynamic preambles "
            "(timestamps, session ids) that defeat slot reuse. "
            "Default: 0 (disabled)."
        ),
    )
    serve.add_argument(
        "--debug-request-dir",
        default=None,
        help=(
            "Directory to dump the full decoded prompt of every "
            "incoming request for offline diffing. Creates one file "
            "per request. Default: disabled."
        ),
    )
    serve.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def run_server(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.command != "serve":
        logger.error("Unknown command: %s", args.command)
        return 2

    kiv_config = KIVConfig(
        hot_budget=args.hot_budget,
        top_p=args.top_p_kiv,
        page_size=args.page_size,
        top_pages=args.top_pages,
    )

    from .model_loader import load_model

    loaded = load_model(
        args.model,
        quantize=args.quantize,
        device_map=args.device_map,
        dtype=args.dtype,
        display_name=args.name,
        trust_remote_code=args.trust_remote_code,
        kiv_config=kiv_config,
        attn_implementation=args.attn_impl,
    )

    logger.info(
        "KIV server ready: model=%s name=%s host=%s port=%d",
        loaded.repo_id, loaded.display_name, args.host, args.port,
    )

    import uvicorn

    from .app import create_app

    prefill_hot_cap = args.prefill_hot_cap if args.prefill_hot_cap > 0 else None
    app = create_app(
        loaded,
        max_slots=args.max_slots,
        prefill_chunk_size=args.prefill_chunk_size,
        prefill_hot_cap=prefill_hot_cap,
        debug_head_tokens=args.debug_head_tokens,
        debug_request_dir=args.debug_request_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def main() -> None:
    sys.exit(run_server())


if __name__ == "__main__":
    main()
