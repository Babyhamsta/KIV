"""Profile KIV decode step to find bottlenecks."""
import sys
import os
import time
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    import torch
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=== KIV Profiler ===", flush=True)
    print("Loading model...", flush=True)

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
    print("Loaded.\n", flush=True)

    from kiv import KIVConfig, KIVMiddleware
    from kiv.eval_utils import FILLER_PARAGRAPHS

    # Build prompts at different lengths
    filler = "\n\n".join(FILLER_PARAGRAPHS * 10)
    question = "What is the capital of France?"

    def make_prompt(target_tokens):
        content = filler + "\n\n" + question
        messages = [{"role": "user", "content": content}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(fmt, return_tensors="pt", max_length=target_tokens, truncation=True)["input_ids"]
        return ids.to(device)

    CHUNK_SIZE = 4096

    def profile_section(name, fn):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  {name:40s} {elapsed:>8.3f}s  {peak:>7.0f} MB", flush=True)
        return result, elapsed

    # Test at multiple context lengths
    for ctx_len in [4096, 8192, 16384, 32768]:
        input_ids = make_prompt(ctx_len)
        actual_len = input_ids.shape[1]

        print(f"\n{'=' * 70}", flush=True)
        print(f"Context: {actual_len} tokens (target {ctx_len})", flush=True)
        print(f"{'Phase':40s} {'Time':>8s}  {'Peak MB':>7s}", flush=True)
        print("-" * 60, flush=True)

        for top_p in [256]:
            config = KIVConfig(hot_budget=2048, top_p=top_p)
            middleware = KIVMiddleware(model, config)
            middleware.install()
            topology = middleware.topology

            cache = middleware.create_cache(device)
            num_chunks = (actual_len + CHUNK_SIZE - 1) // CHUNK_SIZE

            chunk_times = []
            last_logits = None
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, actual_len)
                chunk_ids = input_ids[:, start:end]
                chunk_len = end - start
                cache_len = cache.get_seq_length() if i > 0 else 0

                def run_chunk(cids=chunk_ids):
                    with torch.no_grad():
                        return model(input_ids=cids, past_key_values=cache, use_cache=True)

                outputs, elapsed = profile_section(
                    f"Prefill chunk {i+1}/{num_chunks} ({chunk_len}tok, cache={cache_len})",
                    run_chunk,
                )
                chunk_times.append(elapsed)
                if end == actual_len:
                    last_logits = outputs.logits[:, -1, :]
                del outputs

            def do_eviction():
                cache.mark_prefill_complete()

            _, evict_time = profile_section("Eviction (hot->cold)", do_eviction)

            # Report cold store stats
            for layer_idx in topology.independent_kv_layers:
                cs = cache.cold_stores[layer_idx]
                if cs.cold_length > 0:
                    print(f"    Layer {layer_idx}: cold={cs.cold_length} tokens, "
                          f"pages={cs.num_pages}, finalized={cs._num_finalized}",
                          flush=True)

            decode_times = []
            for step in range(5):
                next_token = last_logits.argmax(dim=-1, keepdim=True)

                def run_decode(nt=next_token):
                    with torch.no_grad():
                        return model(input_ids=nt, past_key_values=cache, use_cache=True)

                outputs, elapsed = profile_section(
                    f"Decode step {step+1} (1 tok, P={top_p})",
                    run_decode,
                )
                decode_times.append(elapsed)
                last_logits = outputs.logits[:, -1, :]
                del outputs

            print(f"\n  --- Decode step breakdown (step 6) ---", flush=True)
            next_token = last_logits.argmax(dim=-1, keepdim=True)

            # Time the full decode
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t_full_start = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            t_full = time.perf_counter() - t_full_start
            print(f"  {'Full decode step':40s} {t_full:>8.3f}s", flush=True)

            # Now profile cold store operations separately
            for layer_idx in topology.independent_kv_layers:
                cs = cache.cold_stores[layer_idx]
                if cs.cold_length > 0:
                    # Fake query for profiling
                    fake_q = torch.randn(
                        1,
                        topology.num_query_heads,
                        1,
                        topology.head_dim,
                        device=device,
                        dtype=torch.bfloat16,
                    )

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    cold_k, cold_v = cs.retrieve_top_kv(fake_q, 1.0, config)
                    torch.cuda.synchronize()
                    t_retrieve = time.perf_counter() - t0

                    print(f"  Layer {layer_idx} cold ops (cold={cs.cold_length}):", flush=True)
                    print(f"    {'retrieve_top_kv':38s} {t_retrieve*1000:>8.2f}ms", flush=True)

                    del cold_k, cold_v, fake_q

            # Summary
            total_prefill = sum(chunk_times)
            avg_decode = sum(decode_times) / len(decode_times)
            print(f"\n  SUMMARY:", flush=True)
            print(f"    Total prefill ({num_chunks} chunks): {total_prefill:.2f}s", flush=True)
            print(f"    Eviction: {evict_time*1000:.1f}ms", flush=True)
            print(f"    Avg decode step: {avg_decode*1000:.1f}ms", flush=True)
            print(f"    Decode tok/s: {1/avg_decode:.1f}", flush=True)

            mem = cache.memory_report()
            print(f"    VRAM (hot+k_index): {mem['total_vram_bytes']/1024/1024:.1f}MB", flush=True)
            print(f"    CPU (v_store): {mem['v_store_cpu_bytes']/1024/1024:.1f}MB", flush=True)

            middleware.uninstall()
            del cache
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
