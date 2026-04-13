"""Multi-model KIV compatibility test."""
import sys
import os
import io
import gc
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3.5-mini-instruct",
]


def test_model(model_id):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
    from kiv import KIVConfig, KIVMiddleware
    from kiv.model_topology import detect_topology
    from kiv.eval_utils import build_needle_prompt

    results = {"model": model_id, "topology": None, "tests": []}

    print(f"\n  Loading {model_id}...", flush=True)
    t0 = time.perf_counter()
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.eval()
        device = next(model.parameters()).device
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  SKIP: {e}", flush=True)
        results["tests"].append({"name": "load", "status": "SKIP", "detail": str(e)})
        return results

    print(f"  Detecting topology...", flush=True)
    try:
        topo = detect_topology(model)
        results["topology"] = {
            "family": topo.model_family,
            "layers": topo.num_hidden_layers,
            "global": len(topo.global_layer_indices),
            "independent": len(topo.independent_kv_layers),
            "shared": len(topo.kv_sharing_map),
            "q_heads": topo.num_query_heads,
            "kv_heads": topo.num_kv_heads,
            "head_dim": topo.head_dim,
        }
        t = results["topology"]
        print(
            f"  Topology: family={t['family']}, layers={t['layers']}, "
            f"global={t['global']}, independent={t['independent']}, "
            f"shared={t['shared']}, q_heads={t['q_heads']}, "
            f"kv_heads={t['kv_heads']}, head_dim={t['head_dim']}",
            flush=True,
        )
        results["tests"].append({"name": "topology", "status": "PASS"})
    except Exception as e:
        print(f"  Topology FAIL: {e}", flush=True)
        results["tests"].append({"name": "topology", "status": "FAIL", "detail": str(e)})
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        return results

    print(f"  Installing KIV...", flush=True)
    config = KIVConfig(hot_budget=2048, top_p=256)
    mw = KIVMiddleware(model, config)
    try:
        mw.install()
        results["tests"].append({"name": "install", "status": "PASS"})
    except Exception as e:
        print(f"  Install FAIL: {e}", flush=True)
        results["tests"].append({"name": "install", "status": "FAIL", "detail": str(e)})
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        return results

    print(f"  Short prompt correctness...", flush=True)
    try:
        prompt = "What is the capital of France?"
        messages = [{"role": "user", "content": prompt}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)

        # Vanilla
        mw.uninstall()
        v_cache = DynamicCache(config=model.config)
        with torch.no_grad():
            v_out = model(input_ids=input_ids, past_key_values=v_cache, use_cache=True)
        v_logits = v_out.logits[0, -1].float()

        # KIV
        mw.install()
        k_cache = mw.create_cache(device)
        with torch.no_grad():
            k_out = model(input_ids=input_ids, past_key_values=k_cache, use_cache=True)
        k_logits = k_out.logits[0, -1].float()

        max_diff = (v_logits - k_logits).abs().max().item()
        cos_sim = F.cosine_similarity(v_logits.unsqueeze(0), k_logits.unsqueeze(0)).item()
        passed = max_diff < 0.01
        status = "PASS" if passed else "FAIL"
        print(f"  Short prompt: {status} (max_diff={max_diff:.6f}, cos={cos_sim:.8f})", flush=True)
        results["tests"].append({
            "name": "short_prompt",
            "status": status,
            "max_diff": max_diff,
            "cos_sim": cos_sim,
        })

        del v_cache, k_cache, v_out, k_out
    except Exception as e:
        print(f"  Short prompt FAIL: {e}", flush=True)
        results["tests"].append({"name": "short_prompt", "status": "FAIL", "detail": str(e)})

    gc.collect(); torch.cuda.empty_cache()

    print(f"  Generation test...", flush=True)
    try:
        cache = mw.create_cache(device)
        messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)

        generated = []
        last_logits = out.logits[:, -1, :]
        for _ in range(20):
            nt = last_logits.argmax(dim=-1, keepdim=True)
            generated.append(nt)
            if nt.item() == tokenizer.eos_token_id:
                break
            with torch.no_grad():
                outputs = model(input_ids=nt, past_key_values=cache, use_cache=True)
            last_logits = outputs.logits[:, -1, :]

        resp = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True) if generated else ""
        has_4 = "4" in resp
        status = "PASS" if has_4 else "FAIL"
        safe = resp[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"  Generation: {status} | {safe}", flush=True)
        results["tests"].append({"name": "generation", "status": status, "response": safe})

        del cache, out
    except Exception as e:
        print(f"  Generation FAIL: {e}", flush=True)
        results["tests"].append({"name": "generation", "status": "FAIL", "detail": str(e)})

    gc.collect(); torch.cuda.empty_cache()

    print(f"  Needle test (4K context)...", flush=True)
    try:
        needle = "The secret password is BLUE ELEPHANT 42."
        question = "Based on the document above, what is the secret password?"
        prompt_obj = build_needle_prompt(tokenizer, 4096, 0.5, needle, question)
        inp = prompt_obj.input_ids.to(device)

        cache = mw.create_cache(device)
        last_logits = mw.chunked_prefill(inp, cache, chunk_size=4096)
        cold = max((s.cold_length for s in cache.cold_stores.values()), default=0)

        generated = []
        for _ in range(30):
            nt = last_logits.argmax(dim=-1, keepdim=True)
            generated.append(nt)
            if nt.item() == tokenizer.eos_token_id:
                break
            with torch.no_grad():
                outputs = model(input_ids=nt, past_key_values=cache, use_cache=True)
            last_logits = outputs.logits[:, -1, :]

        resp = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True) if generated else ""
        hit = "BLUE ELEPHANT 42" in resp.upper()
        status = "PASS" if hit else "FAIL"
        safe = resp[:80].encode("ascii", errors="replace").decode("ascii")
        print(f"  Needle 4K: {status} (cold={cold}) | {safe}", flush=True)
        results["tests"].append({
            "name": "needle_4k",
            "status": status,
            "cold": cold,
            "response": safe,
        })

        del cache
    except Exception as e:
        print(f"  Needle FAIL: {e}", flush=True)
        results["tests"].append({"name": "needle_4k", "status": "FAIL", "detail": str(e)})

    mw.uninstall()
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    return results


def main():
    print("=" * 90)
    print("KIV Multi-Model Compatibility Test")
    print("=" * 90)
    print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

    all_results = []
    for model_id in MODELS:
        print(f"\n{'=' * 90}")
        print(f"MODEL: {model_id}")
        print("=" * 90)
        r = test_model(model_id)
        all_results.append(r)

    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print("=" * 90)
    print(f"\n{'Model':<45} {'Topology':>8} {'Install':>8} {'Short':>8} {'Gen':>8} {'Needle':>8}")
    print("-" * 95)
    for r in all_results:
        tests = {t["name"]: t["status"] for t in r["tests"]}
        print(
            f"{r['model']:<45} "
            f"{tests.get('topology', '-'):>8} "
            f"{tests.get('install', '-'):>8} "
            f"{tests.get('short_prompt', '-'):>8} "
            f"{tests.get('generation', '-'):>8} "
            f"{tests.get('needle_4k', '-'):>8}"
        )

    print(f"\nFinished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
