"""KIV needle grid test: P vs context length vs depth."""
import sys
import os
import time
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

    print("=== KIV Needle Grid Test ===")
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    print("Loading model...", flush=True)
    t0 = time.perf_counter()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    model.eval()
    device = next(model.parameters()).device

    print(f"Model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    from kiv import KIVConfig, KIVMiddleware
    from kiv.tiered_cache import TieredKVCache
    from kiv.eval_utils import build_needle_prompt

    needle = "The secret password is BLUE ELEPHANT 42."
    question = "Based on the document above, what is the secret password?"

    # Vanilla limited to 8K (eager attention OOMs at 16K+).
    # KIV uses chunked prefill (4096-token chunks) so it can go to 32K+.
    VANILLA_MAX_CTX = 8192
    CHUNK_SIZE = 4096

    vanilla_lengths = [4096, 8192]
    kiv_lengths = [4096, 8192, 16384, 32768]
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    p_values = [256, 512, 1024]

    total_tests = (
        len(vanilla_lengths) * len(depths)
        + len(kiv_lengths) * len(depths) * len(p_values)
    )
    test_num = 0

    def log_progress(phase, ctx_len, depth, status, extra=""):
        nonlocal test_num
        test_num += 1
        ts = time.strftime("%H:%M:%S")
        pct = test_num / total_tests * 100
        print(
            f"[{ts}] [{test_num:2d}/{total_tests}] ({pct:5.1f}%) "
            f"{phase:12s} ctx={ctx_len:6d} depth={depth:>4.0%}: {status:4s}{extra}",
            flush=True,
        )

    def run_needle_vanilla(ctx_len, depth):
        """Run vanilla model needle test."""
        try:
            prompt = build_needle_prompt(tokenizer, ctx_len, depth, needle, question)
        except ValueError as e:
            return None, str(e)

        inp = prompt.input_ids.to(device)
        mask = prompt.attention_mask.to(device)

        gc.collect()
        torch.cuda.empty_cache()

        try:
            cache = DynamicCache(config=model.config)
            with torch.no_grad():
                out = model.generate(
                    input_ids=inp, attention_mask=mask,
                    past_key_values=cache, use_cache=True,
                    max_new_tokens=30, do_sample=False,
                )
            resp = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
            hit = "BLUE ELEPHANT 42" in resp.upper()
            return hit, resp[:80].encode("ascii", errors="replace").decode("ascii")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            gc.collect()
            return None, f"OOM: {e}"

    def run_needle_kiv(ctx_len, depth, top_p, middleware):
        """Run KIV needle test with chunked prefill + generate."""
        try:
            prompt = build_needle_prompt(tokenizer, ctx_len, depth, needle, question)
        except ValueError as e:
            return None, str(e), {}

        inp = prompt.input_ids.to(device)

        gc.collect()
        torch.cuda.empty_cache()

        try:
            cache = middleware.create_cache(device)

            # Chunked prefill (avoids quadratic OOM at 16K+)
            next_logits = middleware.chunked_prefill(inp, cache, chunk_size=CHUNK_SIZE)

            # Generate
            generated = []
            for _ in range(30):
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                with torch.no_grad():
                    outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
                next_logits = outputs.logits[:, -1, :]

            if generated:
                gen_ids = torch.cat(generated, dim=-1)
                resp = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            else:
                resp = ""

            hit = "BLUE ELEPHANT 42" in resp.upper()
            mem = cache.memory_report()
            safe = resp[:80].encode("ascii", errors="replace").decode("ascii")
            return hit, safe, mem
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            gc.collect()
            return None, f"OOM: {e}", {}

    print(f"\n{'=' * 90}", flush=True)
    print(f"PHASE 1: Vanilla baseline ({len(vanilla_lengths) * len(depths)} tests, max ctx={VANILLA_MAX_CTX})", flush=True)
    print(f"{'=' * 90}", flush=True)

    vanilla_results = {}
    phase_t0 = time.perf_counter()

    for ctx_len in vanilla_lengths:
        for depth in depths:
            t_start = time.perf_counter()
            hit, resp = run_needle_vanilla(ctx_len, depth)
            elapsed = time.perf_counter() - t_start
            status = "PASS" if hit else ("OOM" if hit is None else "FAIL")
            vanilla_results[(ctx_len, depth)] = hit
            log_progress("Vanilla", ctx_len, depth, status, f" ({elapsed:.1f}s) | {resp[:50]}")

    phase_elapsed = time.perf_counter() - phase_t0
    print(f"  Phase 1 done in {phase_elapsed:.0f}s", flush=True)

    kiv_results = {}  # (p, ctx_len, depth) -> hit

    for top_p in p_values:
        print(f"\n{'=' * 90}", flush=True)
        print(f"PHASE 2: KIV P={top_p} ({len(kiv_lengths) * len(depths)} tests, chunked prefill={CHUNK_SIZE})", flush=True)
        print(f"{'=' * 90}", flush=True)

        phase_t0 = time.perf_counter()
        config = KIVConfig(hot_budget=2048, top_p=top_p)
        middleware = KIVMiddleware(model, config)
        middleware.install()

        for ctx_len in kiv_lengths:
            for depth in depths:
                t_start = time.perf_counter()
                hit, resp, mem = run_needle_kiv(ctx_len, depth, top_p, middleware)
                elapsed = time.perf_counter() - t_start
                status = "PASS" if hit else ("OOM" if hit is None else "FAIL")
                kiv_results[(top_p, ctx_len, depth)] = hit

                mem_str = ""
                if mem:
                    total_mb = mem.get("total_vram_bytes", 0) / 1024 / 1024
                    cpu_mb = mem.get("v_store_cpu_bytes", 0) / 1024 / 1024
                    mem_str = f" [VRAM={total_mb:.1f}MB CPU={cpu_mb:.1f}MB]"

                log_progress(f"P={top_p}", ctx_len, depth, status, f" ({elapsed:.1f}s){mem_str} | {resp[:40]}")

        phase_elapsed = time.perf_counter() - phase_t0
        print(f"  P={top_p} done in {phase_elapsed:.0f}s", flush=True)
        middleware.uninstall()

    total_elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 90}", flush=True)
    print(f"SUMMARY GRID  (total time: {total_elapsed:.0f}s)", flush=True)
    print(f"{'=' * 90}", flush=True)

    # Use all unique context lengths
    all_ctx = sorted(set(vanilla_lengths + kiv_lengths))

    depth_strs = [f"{d:.0%}" for d in depths]
    col_w = 5 * len(depths) + len(depths) - 1

    header = f"{'ctx':>7} | {'Vanilla':^{col_w}}"
    for p in p_values:
        header += f" | {'P=' + str(p):^{col_w}}"
    print(header)

    sub_header = f"{'':>7} | {' '.join(f'{d:>5}' for d in depth_strs)}"
    for p in p_values:
        sub_header += f" | {' '.join(f'{d:>5}' for d in depth_strs)}"
    print(sub_header)
    print("-" * len(sub_header))

    for ctx_len in all_ctx:
        row = f"{ctx_len:>7} |"

        # Vanilla
        for depth in depths:
            hit = vanilla_results.get((ctx_len, depth))
            if hit is None:
                row += "     -"
            elif hit:
                row += "     Y"
            else:
                row += "     N"

        # Each P
        for p in p_values:
            row += " |"
            for depth in depths:
                hit = kiv_results.get((p, ctx_len, depth))
                if hit is None:
                    row += "     -"
                elif hit:
                    row += "     Y"
                else:
                    row += "     N"

        print(row)

    # Stats
    print(f"\nHot budget: 2048 tokens")

    for p in p_values:
        total = sum(1 for k, v in kiv_results.items() if k[0] == p and v is not None)
        passed = sum(1 for k, v in kiv_results.items() if k[0] == p and v is True)
        if total > 0:
            print(f"P={p}: {passed}/{total} passed ({passed/total:.0%})")

    vanilla_total = sum(1 for v in vanilla_results.values() if v is not None)
    vanilla_passed = sum(1 for v in vanilla_results.values() if v is True)
    if vanilla_total > 0:
        print(f"Vanilla: {vanilla_passed}/{vanilla_total} passed ({vanilla_passed/vanilla_total:.0%})")

    print(f"\nFinished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
