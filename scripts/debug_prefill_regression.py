"""
Debug why prefill regressed from 304s to 1067s at 1M tokens.
Profile what happens inside each chunk during bounded prefill.
"""
import gc
import sys
import time

sys.path.insert(0, ".")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from kiv import KIVConfig, KIVMiddleware


def main():
    print("=== Prefill Regression Debug ===", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", quantization_config=bnb, device_map="auto",
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    model.eval()
    device = next(model.parameters()).device

    # Get 100K tokens
    print("Streaming text...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    texts = []
    total = 0
    for s in ds:
        if len(s.get("text", "")) < 100: continue
        texts.append(s["text"])
        total += len(tokenizer.encode(s["text"], add_special_tokens=False))
        if total >= 110000: break
    full_text = "\n\n".join(texts) + "\n\nSummarize."
    messages = [{"role": "user", "content": full_text}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(fmt, return_tensors="pt", max_length=100000, truncation=True)["input_ids"].to(device)
    ctx_len = full_ids.shape[1]
    print(f"Input: {ctx_len} tokens\n", flush=True)

    CHUNK_SIZE = 4096
    num_chunks = (ctx_len + CHUNK_SIZE - 1) // CHUNK_SIZE

    # ── Test 1: Measure evict_from_hot cost in isolation ──
    print("=== Test 1: evict_from_hot timing ===", flush=True)
    config = KIVConfig(hot_budget=2048, top_p=256, page_size=128, top_pages=32)
    middleware = KIVMiddleware(model, config)
    middleware.install()

    cache = middleware.create_cache(device)
    # Mimic chunked_prefill: suppress cold, forward first, evict after
    cache._suppress_cold = True

    print(f"{'Chunk':>6} {'Forward':>8} {'Evict':>8} {'Total':>8} {'Pages':>6} {'Cold':>7}", flush=True)
    print("-" * 55, flush=True)

    for i in range(min(10, num_chunks)):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, ctx_len)
        chunk_ids = full_ids[:, start:end]

        gc.collect(); torch.cuda.empty_cache()

        # Time forward pass (no eviction during forward)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=chunk_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000

        del outputs

        # Time eviction separately (after forward, like chunked_prefill does)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache._evict_to_cap(CHUNK_SIZE)
        torch.cuda.synchronize()
        evict_ms = (time.perf_counter() - t0) * 1000

        store = cache.cold_stores[4]
        pages = store.num_pages
        cold = store.cold_length

        print(f"{i+1:>4}/{num_chunks} {fwd_ms:>7.1f}ms {evict_ms:>7.1f}ms {fwd_ms+evict_ms:>7.1f}ms {pages:>6} {cold:>7}", flush=True)

    middleware.uninstall()
    del cache

    # ── Test 2: Profile inside evict_from_hot ──
    print(f"\n=== Test 2: Breakdown inside evict_from_hot ===", flush=True)

    from kiv.cold_store import ColdKVStore
    store = ColdKVStore(config, device)

    # Simulate eviction of 2048 tokens (typical per chunk)
    fake_k = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)
    fake_v = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)

    # Warmup
    store.evict_from_hot(fake_k.clone(), fake_v.clone())

    # Timed run
    fake_k = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)
    fake_v = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    store.evict_from_hot(fake_k, fake_v)
    torch.cuda.synchronize()
    total_evict = (time.perf_counter() - t0) * 1000
    print(f"  evict_from_hot(2048 tokens): {total_evict:.1f}ms", flush=True)
    print(f"  Pages after: {store.num_pages}, partial: {store._partial_len}", flush=True)

    # Now profile the inner operations
    print(f"\n  Breakdown of page finalization (2048 tok = 16 pages):", flush=True)

    store2 = ColdKVStore(config, device)
    fake_k = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)
    fake_v = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)

    # Manually step through what evict_from_hot does
    store2._partial_k = fake_k
    store2._partial_v = fake_v

    page_size = config.page_size
    times_summary = []
    times_cpu_transfer = []
    times_cat_cpu = []
    times_trim = []

    while store2._partial_len >= page_size:
        page_k = store2._partial_k[:, :, :page_size, :]
        page_v = store2._partial_v[:, :, :page_size, :]

        # Summary
        torch.cuda.synchronize()
        t = time.perf_counter()
        summary = page_k.mean(dim=2, keepdim=True)
        if store2._page_summaries is None:
            store2._page_summaries = summary
        else:
            store2._page_summaries = torch.cat([store2._page_summaries, summary], dim=2)
        torch.cuda.synchronize()
        times_summary.append((time.perf_counter() - t) * 1000)

        # CPU transfer
        t = time.perf_counter()
        k_cpu = page_k.cpu().pin_memory()
        v_cpu = page_v.cpu().pin_memory()
        times_cpu_transfer.append((time.perf_counter() - t) * 1000)

        # Cat on CPU
        t = time.perf_counter()
        if store2._k_pages is None:
            store2._k_pages = k_cpu
            store2._v_store = v_cpu
        else:
            store2._k_pages = torch.cat([store2._k_pages, k_cpu], dim=2)
            store2._v_store = torch.cat([store2._v_store, v_cpu], dim=2)
        times_cat_cpu.append((time.perf_counter() - t) * 1000)

        store2._num_finalized += page_size

        # Trim partial
        t = time.perf_counter()
        store2._partial_k = store2._partial_k[:, :, page_size:, :].contiguous()
        store2._partial_v = store2._partial_v[:, :, page_size:, :].contiguous()
        torch.cuda.synchronize()
        times_trim.append((time.perf_counter() - t) * 1000)

    print(f"    Summary (mean K + cat): avg={sum(times_summary)/len(times_summary):.2f}ms total={sum(times_summary):.1f}ms", flush=True)
    print(f"    CPU transfer (.cpu().pin_memory()): avg={sum(times_cpu_transfer)/len(times_cpu_transfer):.2f}ms total={sum(times_cpu_transfer):.1f}ms", flush=True)
    print(f"    CPU cat: avg={sum(times_cat_cpu)/len(times_cat_cpu):.2f}ms total={sum(times_cat_cpu):.1f}ms", flush=True)
    print(f"    Partial trim (.contiguous()): avg={sum(times_trim)/len(times_trim):.2f}ms total={sum(times_trim):.1f}ms", flush=True)
    print(f"    Total across {len(times_summary)} pages: {sum(times_summary)+sum(times_cpu_transfer)+sum(times_cat_cpu)+sum(times_trim):.1f}ms", flush=True)

    # ── Test 3: Compare with old brute-force eviction (no pages) ──
    print(f"\n=== Test 3: Old-style eviction (no page construction) ===", flush=True)

    fake_k = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)
    fake_v = torch.randn(1, 1, 2048, 512, device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Old style: just GPU->CPU transfer + cat
    k_cpu = fake_k.to("cpu", non_blocking=True)
    v_cpu = fake_v.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    old_transfer = (time.perf_counter() - t0) * 1000
    print(f"  Old eviction (GPU->CPU sync): {old_transfer:.1f}ms", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
