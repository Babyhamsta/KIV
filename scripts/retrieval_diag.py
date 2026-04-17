"""Retrieval hit-rate diagnostic for KIV.

Plants a needle at a known token position inside a long haystack,
queries for it, and reports whether the needle's token range was
pulled back by the top-P retrieval during each decode step.

In contrast to ``needle_grid.py``, this does not score the model's
final answer. It isolates the retrieval stage so retrieval quality
can be analyzed independently of model reasoning quality.

Usage::

    python scripts/retrieval_diag.py \\
        --model google/gemma-4-E2B-it \\
        --context-lengths 32000 128000 512000 1000000 \\
        --top-p 256 1024 \\
        --depth 0.02 \\
        --quantize 4bit

The model is loaded once at startup and reused across every
(context, top_p) configuration so the full sweep stays time-bounded.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--quantize", default="4bit", choices=("4bit", "8bit", "none"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[32000, 128000, 512000, 1000000],
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=0.02,
        help="Fraction into the haystack where the needle goes",
    )
    parser.add_argument(
        "--needle",
        default="The secret codeword that Jace whispered to the robot was BUTTERFLY_17.",
    )
    parser.add_argument(
        "--question",
        default="What was the secret codeword that Jace whispered?",
    )
    parser.add_argument(
        "--top-p",
        type=int,
        nargs="+",
        default=[256, 1024],
    )
    parser.add_argument("--top-pages", type=int, default=32)
    parser.add_argument("--hot-budget", type=int, default=2048)
    parser.add_argument("--prefill-hot-cap", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--num-decode-steps", type=int, default=8)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())

    import torch

    from kiv.config import KIVConfig
    from kiv.eval_utils import build_needle_prompt
    from kiv.server.model_loader import load_model

    print(f"Loading {args.model} ...", flush=True)
    t_load = time.perf_counter()
    quant = None if args.quantize == "none" else args.quantize
    loaded = load_model(
        args.model,
        quantize=quant,
        dtype=args.dtype,
        kiv_config=KIVConfig(
            hot_budget=args.hot_budget,
            top_p=max(args.top_p),
            top_pages=args.top_pages,
        ),
    )
    print(f"  loaded in {time.perf_counter() - t_load:.1f}s\n", flush=True)

    model = loaded.model
    tokenizer = loaded.tokenizer
    middleware = loaded.middleware
    device = next(model.parameters()).device
    needle_len = len(
        tokenizer(args.needle, add_special_tokens=False)["input_ids"]
    )

    rows: list[dict] = []

    for context_length in args.context_lengths:
        print(f"--- context = {context_length} tokens ---", flush=True)
        try:
            prompt = build_needle_prompt(
                tokenizer,
                context_length=context_length,
                depth=args.depth,
                needle=args.needle,
                question=args.question,
            )
        except ValueError as exc:
            print(f"  skip: {exc}", flush=True)
            continue

        input_ids = prompt.input_ids.to(device)
        T = input_ids.shape[1]
        needle_start = prompt.needle_start
        needle_range = set(range(needle_start, needle_start + needle_len))
        print(
            f"  {T} tokens total, needle at [{needle_start}, "
            f"{needle_start + needle_len}) ({needle_len} tok, "
            f"depth {prompt.actual_depth:.2%})",
            flush=True,
        )

        for top_p in args.top_p:
            middleware.config.top_p = top_p

            # Fresh cache per (context, top_p) so the cold store starts clean.
            cache = middleware.create_cache(device)
            t_prefill = time.perf_counter()
            last_logits = middleware.chunked_prefill(
                input_ids,
                cache,
                chunk_size=args.chunk_size,
                prefill_hot_cap=args.prefill_hot_cap,
            )
            prefill_s = time.perf_counter() - t_prefill

            # Clear telemetry so we only measure decode-step retrievals.
            for cs in cache.cold_stores.values():
                cs.reset_telemetry()

            generated = []
            next_logits = last_logits
            with torch.no_grad():
                for _ in range(args.num_decode_steps):
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated.append(int(next_token.item()))
                    outputs = model(
                        input_ids=next_token,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    next_logits = outputs.logits[:, -1, :]

            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

            coarse_hit = 0
            fine_hit = 0
            total_fine = 0
            layer_count = len(cache.cold_stores)
            hit_layers = 0
            for cs in cache.cold_stores.values():
                snap = cs.telemetry_snapshot()
                layer_had_hit = False
                for rec in snap["recent"]:
                    if rec["kind"] == "coarse":
                        ps = rec["pages_selected"]
                        covered = set()
                        for p in ps:
                            covered.update(
                                range(p * 128, (p + 1) * 128)
                            )
                        if covered & needle_range:
                            coarse_hit += 1
                    elif rec["kind"] == "fine":
                        total_fine += 1
                        sel = set(rec["selected_token_indices"])
                        if sel & needle_range:
                            fine_hit += 1
                            layer_had_hit = True
                if layer_had_hit:
                    hit_layers += 1

            coarse_rate = coarse_hit / total_fine if total_fine else 0.0
            fine_rate = fine_hit / total_fine if total_fine else 0.0
            layer_rate = hit_layers / layer_count if layer_count else 0.0

            rows.append({
                "context": T,
                "top_p": top_p,
                "needle_tokens": needle_len,
                "prefill_s": prefill_s,
                "coarse_rate": coarse_rate,
                "fine_rate": fine_rate,
                "layer_rate": layer_rate,
                "decoded": decoded[:80],
            })

            print(
                f"  top_p={top_p:>4}  prefill={prefill_s:>6.1f}s  "
                f"coarse={coarse_rate:>5.1%}  fine={fine_rate:>6.1%}  "
                f"layers={layer_rate:>5.1%}  decoded={decoded[:60]!r}",
                flush=True,
            )

            # Free per-run state before the next config so 1M prefills
            # don't stack CPU RAM across iterations.
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== summary ===", flush=True)
    print(
        f"{'context':>8} {'top_p':>6} {'needle':>6} {'prefill':>8} "
        f"{'coarse':>7} {'fine':>7} {'layers':>7}  answer",
        flush=True,
    )
    for r in rows:
        print(
            f"{r['context']:>8} {r['top_p']:>6} {r['needle_tokens']:>6} "
            f"{r['prefill_s']:>7.1f}s {r['coarse_rate']:>6.1%} "
            f"{r['fine_rate']:>6.1%} {r['layer_rate']:>6.1%}  "
            f"{r['decoded']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
