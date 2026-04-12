"""
KIV scaling profile: prefill, decode, and cold overhead vs context length.
Single model load. Real text from FineWeb-Edu.
Tests 4K -> 100K+ in one pass.
"""
import sys
import os
import time
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset

    print("=== KIV Scaling Profile ===")
    print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

    # ── Load model once ──
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
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    from kiv import KIVConfig, KIVMiddleware

    CHUNK_SIZE = 4096
    P = 256
    HOT_BUDGET = 2048
    NUM_DECODE_STEPS = 10  # enough to measure, not wasteful

    # ── Build large token pool from FineWeb ──
    print("Streaming text from FineWeb-Edu...", flush=True)
    TARGET_TOKENS = 1_100_000  # enough for 1M context with overhead

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    texts = []
    total_tokens = 0
    for sample in ds:
        text = sample.get("text", "")
        if len(text) < 100:
            continue
        texts.append(text)
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
        if total_tokens >= TARGET_TOKENS:
            break

    full_text = "\n\n".join(texts)
    print(f"Collected {total_tokens} tokens from {len(texts)} documents", flush=True)

    # Pre-tokenize the full pool
    question = "\n\nBased on everything above, what are the key themes discussed?"
    messages = [{"role": "user", "content": full_text + question}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)
    max_available = full_ids.shape[1]
    print(f"Full tokenized pool: {max_available} tokens\n", flush=True)

    # ── Install middleware once ──
    config = KIVConfig(hot_budget=HOT_BUDGET, top_p=P)
    middleware = KIVMiddleware(model, config)
    middleware.install()

    # ── Scaling test ──
    context_lengths = [4096, 32768, 100000, 250000, 500000, 1000000]
    PREFILL_HOT_CAP = 4096  # bounded prefill to keep VRAM stable
    results = []

    print(f"{'Ctx':>8} {'Chunks':>6} {'Prefill':>8} {'Prefill':>10} {'Evict':>7} "
          f"{'Decode':>8} {'Cold/L':>7} {'K-score':>8} {'TopP':>6} {'V-fetch':>8} "
          f"{'tok/s':>6} {'HotMB':>6} {'CpuMB':>7}", flush=True)
    print(f"{'':>8} {'':>6} {'(s)':>8} {'(ms/chunk)':>10} {'(ms)':>7} "
          f"{'(ms)':>8} {'(ms)':>7} {'(ms)':>8} {'(ms)':>6} {'(ms)':>8} "
          f"{'':>6} {'VRAM':>6} {'K+V':>7}", flush=True)
    print("-" * 120, flush=True)

    for ctx_len in context_lengths:
        if ctx_len > max_available:
            print(f"{ctx_len:>8}  SKIP — only {max_available} tokens available", flush=True)
            continue

        input_ids = full_ids[:, :ctx_len]
        num_chunks = (ctx_len + CHUNK_SIZE - 1) // CHUNK_SIZE

        gc.collect()
        torch.cuda.empty_cache()

        # ── Prefill (bounded, cap=4K to keep VRAM stable at any context length) ──
        cache = middleware.create_cache(device)

        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        last_logits = middleware.chunked_prefill(
            input_ids, cache, chunk_size=CHUNK_SIZE, prefill_hot_cap=PREFILL_HOT_CAP,
        )
        torch.cuda.synchronize()
        total_prefill = time.perf_counter() - t_prefill_start
        avg_chunk = total_prefill / num_chunks * 1000
        evict_ms = 0  # eviction happens inside chunked_prefill

        # ── Decode steps ──
        decode_times = []
        for step in range(NUM_DECODE_STEPS):
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t_start)
            last_logits = outputs.logits[:, -1, :]
            del outputs

        avg_decode_ms = sum(decode_times) / len(decode_times) * 1000
        tok_per_s = 1000 / avg_decode_ms

        # ── Cold store breakdown ──
        cold_overhead_ms = 0
        k_score_ms = 0
        topk_ms = 0
        v_fetch_ms = 0
        num_cold_layers = 0

        for layer_idx in middleware.topology.independent_kv_layers:
            cs = cache.cold_stores[layer_idx]
            if cs.cold_length > 0:
                num_cold_layers += 1
                fake_q = torch.randn(
                    1, middleware.topology.num_query_heads, 1,
                    middleware.topology.head_dim,
                    device=device, dtype=torch.bfloat16,
                )

                # Profile the full coarse-to-fine pipeline
                torch.cuda.synchronize()
                t = time.perf_counter()
                top_scores, V_fetched = cs.score_and_fetch_cold(fake_q, 1.0, config)
                torch.cuda.synchronize()
                total_cold = (time.perf_counter() - t) * 1000
                cold_overhead_ms += total_cold
                k_score_ms += total_cold  # coarse+fine combined
                # topk and V fetch are now inside score_and_fetch_cold

                del top_scores, V_fetched, fake_q

        cold_per_layer = cold_overhead_ms / max(num_cold_layers, 1)

        # ── Memory ──
        mem = cache.memory_report()
        hot_mb = mem["hot_vram_bytes"] / 1024 / 1024
        cpu_mb = mem["total_cpu_bytes"] / 1024 / 1024

        cold_len = next(iter(cache.cold_stores.values())).cold_length if cache.cold_stores else 0

        print(
            f"{ctx_len:>8} {num_chunks:>6} {total_prefill:>8.2f} {avg_chunk:>10.1f} "
            f"{evict_ms:>7.1f} {avg_decode_ms:>8.1f} {cold_per_layer:>7.2f} "
            f"{k_score_ms/max(num_cold_layers,1):>8.2f} "
            f"{topk_ms/max(num_cold_layers,1):>6.2f} "
            f"{v_fetch_ms/max(num_cold_layers,1):>8.2f} "
            f"{tok_per_s:>6.1f} {hot_mb:>6.1f} {cpu_mb:>7.1f}",
            flush=True,
        )

        results.append({
            "ctx": ctx_len, "chunks": num_chunks,
            "prefill_s": total_prefill, "chunk_ms": avg_chunk,
            "evict_ms": evict_ms, "decode_ms": avg_decode_ms,
            "cold_per_layer_ms": cold_per_layer,
            "k_score_ms": k_score_ms / max(num_cold_layers, 1),
            "topk_ms": topk_ms / max(num_cold_layers, 1),
            "v_fetch_ms": v_fetch_ms / max(num_cold_layers, 1),
            "tok_s": tok_per_s, "hot_mb": hot_mb, "cpu_mb": cpu_mb,
            "cold_tokens": cold_len,
        })

        del cache
        gc.collect()
        torch.cuda.empty_cache()

    middleware.uninstall()

    # ── Summary ──
    print(f"\n{'=' * 120}", flush=True)
    print(f"Config: hot_budget={HOT_BUDGET}, P={P}, chunk_size={CHUNK_SIZE}", flush=True)
    print(f"Decode steps per test: {NUM_DECODE_STEPS}", flush=True)

    print(f"\nScaling summary:", flush=True)
    print(f"{'Context':>8} {'Cold tok':>9} {'Prefill':>8} {'Decode':>8} {'Cold OH':>8} {'KIV %':>6} {'Hot MB':>7} {'CPU MB':>8}", flush=True)
    print("-" * 75, flush=True)
    for r in results:
        kiv_pct = r["cold_per_layer_ms"] * 3 / r["decode_ms"] * 100 if r["decode_ms"] > 0 else 0
        print(
            f"{r['ctx']:>8} {r['cold_tokens']:>9} {r['prefill_s']:>7.2f}s "
            f"{r['decode_ms']:>7.1f}ms {r['cold_per_layer_ms']*3:>7.2f}ms "
            f"{kiv_pct:>5.1f}% {r['hot_mb']:>6.1f}MB {r['cpu_mb']:>7.1f}MB",
            flush=True,
        )

    total = time.perf_counter() - t0
    print(f"\nTotal time: {total:.0f}s", flush=True)
    print(f"Finished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
