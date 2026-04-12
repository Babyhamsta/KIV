"""
Evaluation harness for KIV middleware.

Tests: correctness baseline, compression quality, needle retrieval, memory.
"""

from __future__ import annotations

import gc
import logging
import sys
import time

import torch
import torch.nn.functional as F

from .config import KIVConfig
from .middleware import KIVMiddleware
from .tiered_cache import TieredKVCache

logger = logging.getLogger(__name__)


def _generate_with_kiv(
    model, tokenizer, middleware: KIVMiddleware,
    input_ids: torch.Tensor, max_new_tokens: int = 50,
) -> tuple[torch.Tensor, TieredKVCache]:
    """Generate tokens using KIV middleware with manual prefill/generate split."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    cache = middleware.create_cache(device)

    # Prefill: run full prompt through model (no eviction yet)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )

    # Mark prefill complete — triggers eviction of excess tokens
    cache.mark_prefill_complete()

    # Generate token by token
    next_token_logits = outputs.logits[:, -1, :]
    generated = []

    for _ in range(max_new_tokens):
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)

        if next_token.item() == tokenizer.eos_token_id:
            break

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
            )
        next_token_logits = outputs.logits[:, -1, :]

    if generated:
        gen_ids = torch.cat(generated, dim=-1)
        full_ids = torch.cat([input_ids, gen_ids], dim=-1)
    else:
        full_ids = input_ids

    return full_ids, cache


def test_correctness_short(model, tokenizer, middleware: KIVMiddleware) -> bool:
    """
    Test 1: Short prompt (< hot_budget). Output should match vanilla exactly.
    """
    print("=== Test 1: Short prompt correctness ===")
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    seq_len = input_ids.shape[1]
    print(f"  Prompt length: {seq_len} tokens (hot_budget={middleware.config.hot_budget})")

    # Vanilla forward (with standard cache for fair comparison)
    middleware.uninstall()
    from transformers import DynamicCache
    vanilla_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        vanilla_out = model(input_ids=input_ids, past_key_values=vanilla_cache, use_cache=True)
    vanilla_logits = vanilla_out.logits[0, -1].float()

    # KIV forward
    middleware.install()
    cache = middleware.create_cache(device)
    with torch.no_grad():
        kiv_out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    kiv_logits = kiv_out.logits[0, -1].float()

    # Compare
    max_diff = (vanilla_logits - kiv_logits).abs().max().item()
    cos_sim = F.cosine_similarity(vanilla_logits.unsqueeze(0), kiv_logits.unsqueeze(0)).item()

    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    passed = max_diff < 0.01
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_correctness_long(model, tokenizer, middleware: KIVMiddleware) -> dict:
    """
    Test 2: Long prompt (> hot_budget) with P=all.
    Should closely match vanilla since we retrieve all cold entries.
    """
    print("\n=== Test 2: Long prompt with P=all (lossless retrieval) ===")

    # Build a prompt longer than hot_budget
    from .eval_utils import FILLER_PARAGRAPHS

    filler = "\n\n".join(FILLER_PARAGRAPHS * 3)  # ~3K+ tokens
    question = "Summarize the key points from the text above."
    content = filler + "\n\n" + question
    messages = [{"role": "user", "content": content}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt", max_length=4096, truncation=True)["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    seq_len = input_ids.shape[1]
    print(f"  Prompt length: {seq_len} tokens (hot_budget={middleware.config.hot_budget})")

    if seq_len <= middleware.config.hot_budget:
        print("  SKIP: prompt shorter than hot_budget, can't test eviction")
        return {"skipped": True}

    # Vanilla prefill + one decode step
    middleware.uninstall()
    from transformers import DynamicCache
    vanilla_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        vanilla_prefill = model(
            input_ids=input_ids,
            past_key_values=vanilla_cache,
            use_cache=True,
        )
    next_token = vanilla_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        vanilla_out = model(
            input_ids=next_token,
            past_key_values=vanilla_cache,
            use_cache=True,
        )
    vanilla_logits = vanilla_out.logits[0, -1].float()

    # KIV forward with P = all cold tokens (should be lossless)
    orig_top_p = middleware.config.top_p
    orig_top_pages = middleware.config.top_pages
    middleware.config.top_p = 999999  # retrieve everything
    middleware.config.top_pages = 999999
    try:
        middleware.install()
        cache = middleware.create_cache(device)
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        cache.mark_prefill_complete()
        with torch.no_grad():
            kiv_out = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
            )
        kiv_logits = kiv_out.logits[0, -1].float()
    finally:
        middleware.config.top_p = orig_top_p
        middleware.config.top_pages = orig_top_pages

    # Compare
    max_diff = (vanilla_logits - kiv_logits).abs().max().item()
    cos_sim = F.cosine_similarity(vanilla_logits.unsqueeze(0), kiv_logits.unsqueeze(0)).item()

    # KL divergence
    vanilla_probs = F.softmax(vanilla_logits, dim=-1)
    kiv_probs = F.softmax(kiv_logits, dim=-1)
    kl = F.kl_div(kiv_probs.log(), vanilla_probs, reduction="sum").item()

    cold_len = 0
    if cache.cold_stores:
        first_store = next(iter(cache.cold_stores.values()))
        cold_len = first_store.cold_length
    hot_len = seq_len - cold_len

    print(f"  Hot: {hot_len}, Cold: {cold_len}")
    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  KL divergence: {kl:.6f}")

    return {"max_diff": max_diff, "cos_sim": cos_sim, "kl": kl}


def test_compression_quality(model, tokenizer, middleware: KIVMiddleware) -> dict:
    """
    Test 3: Long prompt with P=256 (lossy). Measure quality degradation.
    """
    print(f"\n=== Test 3: Compression quality (P={middleware.config.top_p}) ===")

    from .eval_utils import FILLER_PARAGRAPHS

    filler = "\n\n".join(FILLER_PARAGRAPHS * 3)
    question = "Summarize the key points from the text above."
    content = filler + "\n\n" + question
    messages = [{"role": "user", "content": content}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt", max_length=4096, truncation=True)["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    seq_len = input_ids.shape[1]
    print(f"  Prompt length: {seq_len} tokens")

    if seq_len <= middleware.config.hot_budget:
        print("  SKIP: prompt shorter than hot_budget")
        return {"skipped": True}

    # Vanilla prefill + one decode step
    middleware.uninstall()
    from transformers import DynamicCache
    vanilla_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        vanilla_prefill = model(
            input_ids=input_ids,
            past_key_values=vanilla_cache,
            use_cache=True,
        )
    next_token = vanilla_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        vanilla_out = model(
            input_ids=next_token,
            past_key_values=vanilla_cache,
            use_cache=True,
        )
    vanilla_logits = vanilla_out.logits[0, -1].float()

    # KIV with configured top_p, exercising retrieval on the first decode step
    middleware.install()
    cache = middleware.create_cache(device)
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    cache.mark_prefill_complete()
    with torch.no_grad():
        kiv_out = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
        )
    kiv_logits = kiv_out.logits[0, -1].float()

    # Metrics
    max_diff = (vanilla_logits - kiv_logits).abs().max().item()
    cos_sim = F.cosine_similarity(vanilla_logits.unsqueeze(0), kiv_logits.unsqueeze(0)).item()
    vanilla_probs = F.softmax(vanilla_logits, dim=-1)
    kiv_probs = F.softmax(kiv_logits, dim=-1)
    kl = F.kl_div(kiv_probs.log(), vanilla_probs, reduction="sum").item()

    # Top-5 comparison
    v_top5 = vanilla_logits.topk(5)
    k_top5 = kiv_logits.topk(5)

    print(f"  Max logit diff: {max_diff:.4f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  KL divergence: {kl:.6f}")
    v_tokens = [tokenizer.decode([i]).encode("ascii", errors="replace").decode("ascii") for i in v_top5.indices.tolist()]
    k_tokens = [tokenizer.decode([i]).encode("ascii", errors="replace").decode("ascii") for i in k_top5.indices.tolist()]
    print(f"  Vanilla top-5: {v_tokens}")
    print(f"  KIV top-5:     {k_tokens}")

    mem = cache.memory_report()
    print(f"  Memory: hot={mem['hot_vram_bytes']/1024:.0f}KB, "
          f"k_index_cpu={mem['k_index_cpu_bytes']/1024:.0f}KB, "
          f"v_store_cpu={mem['v_store_cpu_bytes']/1024:.0f}KB")

    return {"max_diff": max_diff, "cos_sim": cos_sim, "kl": kl, "memory": mem}


def test_needle(model, tokenizer, middleware: KIVMiddleware) -> dict:
    """
    Test 4: Needle-in-haystack with KIV vs vanilla.
    """
    print(f"\n=== Test 4: Needle-in-haystack (KIV P={middleware.config.top_p}) ===")

    from .eval_utils import build_needle_prompt

    needle = "The secret password is BLUE ELEPHANT 42."
    question = "Based on the document above, what is the secret password?"

    results = {}
    for ctx_len in [2048, 4096]:
        for depth in [0.25, 0.50, 0.75]:
            # Vanilla
            middleware.uninstall()
            prompt = build_needle_prompt(tokenizer, ctx_len, depth, needle, question)
            inputs = {
                "input_ids": prompt.input_ids.to(next(model.parameters()).device),
                "attention_mask": prompt.attention_mask.to(next(model.parameters()).device),
            }

            from transformers import DynamicCache
            vanilla_cache = DynamicCache(config=model.config)
            with torch.no_grad():
                v_out = model.generate(
                    **inputs, past_key_values=vanilla_cache,
                    use_cache=True, max_new_tokens=30, do_sample=False,
                )
            v_resp = tokenizer.decode(v_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            v_hit = "BLUE ELEPHANT 42" in v_resp.upper()

            # KIV
            middleware.install()
            kiv_ids, kiv_cache = _generate_with_kiv(
                model, tokenizer, middleware,
                prompt.input_ids, max_new_tokens=30,
            )
            k_resp = tokenizer.decode(kiv_ids[0][prompt.input_ids.shape[1]:], skip_special_tokens=True)
            k_hit = "BLUE ELEPHANT 42" in k_resp.upper()

            v_status = "PASS" if v_hit else "FAIL"
            k_status = "PASS" if k_hit else "FAIL"
            safe_v = v_resp[:80].encode("ascii", errors="replace").decode("ascii")
            safe_k = k_resp[:80].encode("ascii", errors="replace").decode("ascii")
            print(f"  ctx={ctx_len} depth={depth:.0%}: vanilla={v_status} kiv={k_status}")
            print(f"    vanilla: {safe_v}")
            print(f"    kiv:     {safe_k}")

            results[(ctx_len, depth)] = {"vanilla": v_hit, "kiv": k_hit}

    return results


def run_all_tests(model, tokenizer, config: KIVConfig | None = None) -> None:
    """Run the full evaluation suite."""
    config = config or KIVConfig()
    middleware = KIVMiddleware(model, config)
    middleware.install()

    print(f"KIV Config: hot_budget={config.hot_budget}, top_p={config.top_p}")
    print()

    test_correctness_short(model, tokenizer, middleware)
    test_correctness_long(model, tokenizer, middleware)
    test_compression_quality(model, tokenizer, middleware)
    test_needle(model, tokenizer, middleware)

    middleware.uninstall()
    print("\n=== All tests complete ===")
