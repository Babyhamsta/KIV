"""KIV decode pipeline profiler using CUDA events."""
import sys
import os
import io
import gc
import time
import functools

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from datasets import load_dataset


def cuda_timer():
    """Create a pair of CUDA events for timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def time_region(start_event, end_event):
    """Get elapsed ms between two recorded CUDA events."""
    end_event.synchronize()
    return start_event.elapsed_time(end_event)


def main():
    print("=" * 70)
    print("KIV Decode Pipeline Profiler (CUDA Events)")
    print("=" * 70)

    print("Loading model (eager)...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    model.eval()
    device = next(model.parameters()).device

    print("Streaming text from FineWeb-Edu...", flush=True)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )
    texts = []
    total = 0
    for sample in ds:
        text = sample.get("text", "")
        if len(text) < 100:
            continue
        texts.append(text)
        total += len(tokenizer.encode(text, add_special_tokens=False))
        if total >= 1_050_000:
            break

    full_text = "\n\n".join(texts) + "\n\nSummarize the key themes."
    messages = [{"role": "user", "content": full_text}]
    fmt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"][:, :1_000_000].to(
        device
    )
    print(f"Context: {input_ids.shape[1]:,} tokens\n", flush=True)

    from kiv import KIVConfig, KIVMiddleware
    from kiv.cold_store import _repeat_kv

    config = KIVConfig(hot_budget=2048, top_p=256)

    print("=" * 70)
    print("1. VANILLA BASELINE (2K context, eager)")
    print("=" * 70)

    v_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=input_ids[:, :2048], past_key_values=v_cache, use_cache=True)

    # Warmup
    nt = torch.tensor([[1]], device=device)
    for _ in range(3):
        with torch.no_grad():
            model(input_ids=nt, past_key_values=v_cache, use_cache=True)

    vanilla_times = []
    for _ in range(30):
        s, e = cuda_timer()
        s.record()
        with torch.no_grad():
            out = model(input_ids=nt, past_key_values=v_cache, use_cache=True)
        e.record()
        vanilla_times.append(time_region(s, e))

    avg_vanilla = sum(vanilla_times[5:]) / len(vanilla_times[5:])  # drop first 5
    print(f"  Vanilla 2K decode: {avg_vanilla:.2f}ms ({1000/avg_vanilla:.1f} tok/s)")

    del v_cache, out
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("2. KIV DECODE (1M context, eager)")
    print("=" * 70)

    mw = KIVMiddleware(model, config)
    mw.install()
    cache = mw.create_cache(device)

    print("  Prefilling 1M tokens...", flush=True)
    t0 = time.perf_counter()
    last_logits = mw.chunked_prefill(
        input_ids, cache, chunk_size=4096, prefill_hot_cap=4096
    )
    print(f"  Prefill done in {time.perf_counter() - t0:.1f}s", flush=True)

    cold_len = max((s.cold_length for s in cache.cold_stores.values()), default=0)
    print(f"  Cold tokens: {cold_len:,}")

    # Warmup
    for _ in range(3):
        nt = last_logits.argmax(dim=-1, keepdim=True)
        with torch.no_grad():
            out = model(input_ids=nt, past_key_values=cache, use_cache=True)
        last_logits = out.logits[:, -1, :]

    # Full decode timing
    full_times = []
    for _ in range(30):
        nt = last_logits.argmax(dim=-1, keepdim=True)
        s, e = cuda_timer()
        s.record()
        with torch.no_grad():
            out = model(input_ids=nt, past_key_values=cache, use_cache=True)
        e.record()
        full_times.append(time_region(s, e))
        last_logits = out.logits[:, -1, :]

    avg_full = sum(full_times[5:]) / len(full_times[5:])
    print(f"\n  Full decode step: {avg_full:.2f}ms ({1000/avg_full:.1f} tok/s)")
    print(f"  KIV overhead vs vanilla: {avg_full - avg_vanilla:.2f}ms")

    print(f"\n{'=' * 70}")
    print("3. COLD RETRIEVAL BREAKDOWN (per independent layer)")
    print("=" * 70)

    topo = mw.topology
    total_cold_ms = 0

    for layer_idx in topo.independent_kv_layers:
        cs = cache.cold_stores[layer_idx]
        fake_q = torch.randn(
            1, topo.num_query_heads, 1, topo.head_dim,
            device=device, dtype=torch.bfloat16,
        )
        cs._materialize()

            coarse_times = []
        for _ in range(20):
            s, e = cuda_timer()
            s.record()
            summaries_exp = _repeat_kv(cs._page_summaries, cs._num_kv_groups)
            coarse_scores = torch.matmul(fake_q, summaries_exp.transpose(-2, -1))
            _, top_pages = coarse_scores.topk(config.top_pages, dim=-1)
            e.record()
            coarse_times.append(time_region(s, e))
            del summaries_exp, coarse_scores

        avg_coarse = sum(coarse_times[5:]) / len(coarse_times[5:])

        full_retrieve_times = []
        for _ in range(20):
            s, e = cuda_timer()
            s.record()
            cold_k, cold_v = cs.retrieve_top_kv(fake_q, 1.0, config)
            e.record()
            full_retrieve_times.append(time_region(s, e))
            del cold_k, cold_v

        avg_retrieve = sum(full_retrieve_times[5:]) / len(full_retrieve_times[5:])
        avg_fine_fetch = avg_retrieve - avg_coarse

        total_cold_ms += avg_retrieve
        print(
            f"  Layer {layer_idx:2d}: "
            f"coarse={avg_coarse:.2f}ms  "
            f"fine+fetch={avg_fine_fetch:.2f}ms  "
            f"total={avg_retrieve:.2f}ms  "
            f"pages={cs.num_pages}"
        )

    print(f"\n  Total cold retrieval (all layers): {total_cold_ms:.2f}ms")
    print(f"  Model forward (non-cold):          {avg_full - total_cold_ms:.2f}ms")

    print(f"\n{'=' * 70}")
    print("4. ATTENTION FUNCTION COST (concat + mask extend)")
    print("=" * 70)

    # Time just the concatenation + mask extension that KIV adds
    # Simulate: cold K/V + hot K/V concat + mask extend
    hot_k = torch.randn(1, topo.num_kv_heads, 2048, topo.head_dim, device=device, dtype=torch.bfloat16)
    hot_v = torch.randn_like(hot_k)
    cold_k = torch.randn(1, topo.num_kv_heads, 256, topo.head_dim, device=device, dtype=torch.bfloat16)
    cold_v = torch.randn_like(cold_k)
    mask = torch.zeros(1, 1, 1, 2048, device=device, dtype=torch.bfloat16)

    from kiv.middleware import _extend_mask_for_cold

    concat_times = []
    for _ in range(100):
        s, e = cuda_timer()
        s.record()
        combined_k = torch.cat([cold_k, hot_k], dim=2)
        combined_v = torch.cat([cold_v, hot_v], dim=2)
        combined_mask = _extend_mask_for_cold(mask, 256)
        e.record()
        concat_times.append(time_region(s, e))
        del combined_k, combined_v, combined_mask

    avg_concat = sum(concat_times[10:]) / len(concat_times[10:])
    print(f"  Concat + mask extend: {avg_concat:.3f}ms (per layer)")
    print(f"  x7 global layers: {avg_concat * 7:.3f}ms")

    del hot_k, hot_v, cold_k, cold_v, mask

    print(f"\n{'=' * 70}")
    print("5. SDPA COMPARISON")
    print("=" * 70)

    mw.uninstall()
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()

    print("  Loading model (sdpa)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    mw2 = KIVMiddleware(model, config)
    mw2.install()
    cache2 = mw2.create_cache(device)

    print("  Prefilling 1M tokens...", flush=True)
    t0 = time.perf_counter()
    last_logits = mw2.chunked_prefill(
        input_ids, cache2, chunk_size=4096, prefill_hot_cap=4096
    )
    print(f"  Prefill done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Warmup
    for _ in range(3):
        nt = last_logits.argmax(dim=-1, keepdim=True)
        with torch.no_grad():
            out = model(input_ids=nt, past_key_values=cache2, use_cache=True)
        last_logits = out.logits[:, -1, :]

    sdpa_times = []
    for _ in range(30):
        nt = last_logits.argmax(dim=-1, keepdim=True)
        s, e = cuda_timer()
        s.record()
        with torch.no_grad():
            out = model(input_ids=nt, past_key_values=cache2, use_cache=True)
        e.record()
        sdpa_times.append(time_region(s, e))
        last_logits = out.logits[:, -1, :]

    avg_sdpa = sum(sdpa_times[5:]) / len(sdpa_times[5:])
    print(f"\n  SDPA decode step: {avg_sdpa:.2f}ms ({1000/avg_sdpa:.1f} tok/s)")

    mw2.uninstall()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Vanilla 2K (eager):        {avg_vanilla:.2f}ms  ({1000/avg_vanilla:.1f} tok/s)")
    print(f"  KIV 1M (eager):            {avg_full:.2f}ms  ({1000/avg_full:.1f} tok/s)")
    print(f"  KIV 1M (sdpa):             {avg_sdpa:.2f}ms  ({1000/avg_sdpa:.1f} tok/s)")
    print(f"")
    print(f"  Breakdown of KIV 1M (eager):")
    print(f"    Model forward (non-cold): {avg_full - total_cold_ms:.2f}ms")
    print(f"    Cold retrieval (3 layers): {total_cold_ms:.2f}ms")
    print(f"    Concat+mask (7 layers):    {avg_concat * 7:.2f}ms")
    print(f"")
    print(f"  Potential savings:")
    print(f"    eager -> sdpa:             {avg_full - avg_sdpa:.2f}ms")
    print(f"    Target 100ms (10 tok/s):   need to cut {avg_full - 100:.0f}ms")
    print(f"    Theoretical floor:         {avg_vanilla:.2f}ms (vanilla 2K)")


if __name__ == "__main__":
    main()
