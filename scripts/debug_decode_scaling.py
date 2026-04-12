"""
Debug why model forward pass slows down with context length.

WARNING: This script uses removed APIs (config.global_layer_indices,
hardcoded range(35) layer counts) and will not run as-is. These fields
now live on middleware.topology (e.g. middleware.topology.global_layer_indices,
middleware.topology.independent_kv_layers). Needs updating for the new
KIV middleware/topology API before use.

With hot_budget=2048 and sliding_window=512, every layer should see
bounded KV during decode. If decode time grows with context, something
inside the model is scaling unexpectedly.

This script profiles per-layer timing during a single decode step
at different context lengths to find the culprit.
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
    print("=== Decode Scaling Debug ===", flush=True)

    # Load model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", quantization_config=bnb, device_map="auto",
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    model.eval()
    device = next(model.parameters()).device
    text_model = model.model.language_model

    # Stream enough text
    print("Streaming text...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    texts = []
    total_tok = 0
    for s in ds:
        if len(s.get("text", "")) < 100:
            continue
        texts.append(s["text"])
        total_tok += len(tokenizer.encode(s["text"], add_special_tokens=False))
        if total_tok >= 550000:
            break

    full_text = "\n\n".join(texts) + "\n\nSummarize the key themes."
    messages = [{"role": "user", "content": full_text}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)
    max_len = full_ids.shape[1]
    print(f"Token pool: {max_len}\n", flush=True)

    config = KIVConfig(hot_budget=2048, top_p=256)
    middleware = KIVMiddleware(model, config)
    middleware.install()

    CHUNK_SIZE = 4096

    for ctx_len in [4096, 32768, 100000, 250000, 500000]:
        if ctx_len > max_len:
            print(f"\n{ctx_len:>8}  SKIP (only {max_len} tokens)", flush=True)
            continue

        gc.collect(); torch.cuda.empty_cache()

        input_ids = full_ids[:, :ctx_len]
        cache = middleware.create_cache(device)

        # Prefill with bounded cap
        middleware.chunked_prefill(input_ids, cache, chunk_size=CHUNK_SIZE, prefill_hot_cap=CHUNK_SIZE)

        # Warmup decode
        next_token = torch.tensor([[1]], device=device)
        with torch.no_grad():
            model(input_ids=next_token, past_key_values=cache, use_cache=True)

        # ── Profile full decode step ──
        next_token = torch.tensor([[1]], device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=next_token, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) * 1000

        # ── Profile per-layer ──
        # Hook each layer to measure time
        layer_times = {}
        hooks = []

        def make_pre_hook(idx):
            def hook(module, args):
                torch.cuda.synchronize()
                layer_times[idx] = -time.perf_counter()
            return hook

        def make_post_hook(idx):
            def hook(module, args, output):
                torch.cuda.synchronize()
                layer_times[idx] += time.perf_counter()
            return hook

        for i, layer in enumerate(text_model.layers):
            hooks.append(layer.register_forward_pre_hook(make_pre_hook(i)))
            hooks.append(layer.register_forward_hook(make_post_hook(i)))

        next_token = torch.tensor([[1]], device=device)
        torch.cuda.synchronize()
        with torch.no_grad():
            model(input_ids=next_token, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()

        for h in hooks:
            h.remove()

        # ── Analyze ──
        cold_len = cache.cold_stores[4].cold_length

        # Categorize layers
        sliding_layers = [i for i in range(35) if i not in config.global_layer_indices]
        global_layers = list(config.global_layer_indices)

        sliding_total = sum(layer_times.get(i, 0) for i in sliding_layers) * 1000
        global_total = sum(layer_times.get(i, 0) for i in global_layers) * 1000
        sliding_avg = sliding_total / len(sliding_layers)
        global_avg = global_total / len(global_layers)

        # Check cache sizes for each layer type
        sliding_kv_lens = []
        global_hot_lens = []
        for i in range(35):
            if i < len(cache.layers) and cache.layers[i].is_initialized:
                kv_len = cache.layers[i].keys.shape[2] if hasattr(cache.layers[i], 'keys') and cache.layers[i].keys.numel() > 0 else 0
                if i in config.global_layer_indices:
                    global_hot_lens.append(kv_len)
                else:
                    sliding_kv_lens.append(kv_len)

        # Check sliding window cumulative_length
        sliding_cumlen = []
        for i in sliding_layers[:5]:
            if i < len(cache.layers):
                layer_cache = cache.layers[i]
                if hasattr(layer_cache, 'cumulative_length'):
                    sliding_cumlen.append(layer_cache.cumulative_length)

        print(f"\nctx={ctx_len:>7} | cold={cold_len:>7} | full decode={full_ms:>7.1f}ms", flush=True)
        print(f"  Sliding layers ({len(sliding_layers)}): total={sliding_total:.1f}ms avg={sliding_avg:.2f}ms", flush=True)
        print(f"  Global layers  ({len(global_layers)}):  total={global_total:.1f}ms avg={global_avg:.2f}ms", flush=True)

        if sliding_kv_lens:
            print(f"  Sliding KV lens: min={min(sliding_kv_lens)} max={max(sliding_kv_lens)}", flush=True)
        if global_hot_lens:
            print(f"  Global hot lens: {global_hot_lens}", flush=True)
        if sliding_cumlen:
            print(f"  Sliding cumulative_length (first 5): {sliding_cumlen}", flush=True)

        # Per-layer breakdown (find outliers)
        sorted_layers = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 5 slowest layers:", flush=True)
        for idx, t in sorted_layers[:5]:
            ltype = "GLOBAL" if idx in config.global_layer_indices else "sliding"
            kv = 0
            if idx < len(cache.layers) and cache.layers[idx].is_initialized:
                kv = cache.layers[idx].keys.shape[2] if hasattr(cache.layers[idx], 'keys') and cache.layers[idx].keys.numel() > 0 else 0
            print(f"    Layer {idx:>2} ({ltype:>7}): {t*1000:>7.2f}ms  kv_len={kv}", flush=True)

        # Check what's happening with the mask
        seq_len_reported = cache.get_seq_length(0)
        print(f"  cache.get_seq_length(0)={seq_len_reported} (sliding layer 0)", flush=True)
        print(f"  cache.get_seq_length(4)={cache.get_seq_length(4)} (global layer 4)", flush=True)

        del cache
        gc.collect(); torch.cuda.empty_cache()

    middleware.uninstall()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
