"""Test bounded prefill quality impact."""
import gc
import time

from _helpers import (
    KIVConfig, KIVMiddleware,
    load_model, make_phonebook, format_phonebook, safe_str,
)


def main():
    import torch
    import torch.nn.functional as F
    from kiv.eval_utils import build_needle_prompt

    print("=== Bounded Prefill Quality Test ===")
    print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

    model, tokenizer, device = load_model(attn_implementation="sdpa")

    P = 256
    CHUNK_SIZE = 4096
    caps = [None, 16384, 8192, 4096]  # None = unbounded

    config = KIVConfig(hot_budget=2048, top_p=P)
    middleware = KIVMiddleware(model, config)
    middleware.install()

    def run_with_cap(text, cap, max_new_tokens=60):
        """Run generation with a specific prefill_hot_cap."""
        messages = [{"role": "user", "content": text}]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)

        gc.collect(); torch.cuda.empty_cache()

        cache = middleware.create_cache(device)
        t0 = time.perf_counter()
        last_logits = middleware.chunked_prefill(
            input_ids, cache, chunk_size=CHUNK_SIZE, prefill_hot_cap=cap,
        )
        prefill_time = time.perf_counter() - t0

        logits_snapshot = last_logits.float().cpu()

        generated = []
        for _ in range(max_new_tokens):
            nt = last_logits.argmax(dim=-1, keepdim=True)
            generated.append(nt)
            if nt.item() == tokenizer.eos_token_id:
                break
            with torch.no_grad():
                outputs = model(input_ids=nt, past_key_values=cache, use_cache=True)
            last_logits = outputs.logits[:, -1, :]

        resp = tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True) if generated else ""
        return resp, logits_snapshot, prefill_time, cache

    print("=" * 90, flush=True)
    print("TEST 1: Phonebook lookup (1000 entries, ~29K tokens)", flush=True)
    print("=" * 90, flush=True)

    entries = make_phonebook(1000)
    phonebook = format_phonebook(entries)
    tok_count = len(tokenizer.encode(phonebook, add_special_tokens=False))
    print(f"  {len(entries)} entries, ~{tok_count} tokens\n", flush=True)

    targets = [entries[250], entries[500], entries[750]]
    cap_labels = {None: "None (full)", 16384: "16K", 8192: "8K", 4096: "4K"}

    print(f"  {'Target':>20} | {'Cap':>12} | {'Hit':>4} | {'Prefill':>8} | {'Response'}", flush=True)
    print(f"  {'-'*85}", flush=True)

    baseline_logits = {}
    for target in targets:
        query = f"What is {target['name']}'s phone number?"
        full_text = f"Here is a phone directory:\n\n{phonebook}\n\n{query}"

        for cap in caps:
            resp, logits, ptime, cache = run_with_cap(full_text, cap)
            hit = target["phone"] in resp

            kl_str = ""
            key = target["name"]
            if cap is None:
                baseline_logits[key] = logits
            elif key in baseline_logits:
                bl = F.softmax(baseline_logits[key][0], dim=-1)
                cl = F.softmax(logits[0], dim=-1)
                kl = F.kl_div(cl.log(), bl, reduction="sum").item()
                kl_str = f" KL={kl:.4f}"

            print(
                f"  {target['name']:>20} | {cap_labels[cap]:>12} | "
                f"{'PASS' if hit else 'FAIL':>4} | {ptime:>7.1f}s |{kl_str} {safe_str(resp, 50)}",
                flush=True,
            )
        print(flush=True)

    print("=" * 90, flush=True)
    print("TEST 2: Needle-in-haystack at 8K context (5 depths)", flush=True)
    print("=" * 90, flush=True)

    needle = "The secret password is BLUE ELEPHANT 42."
    question = "Based on the document above, what is the secret password?"
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]

    print(f"  {'Depth':>6} | {'Cap':>12} | {'Hit':>4} | {'Prefill':>8} | {'KL':>8} | {'Response'}", flush=True)
    print(f"  {'-'*90}", flush=True)

    needle_baseline_logits = {}
    for depth in depths:
        prompt = build_needle_prompt(tokenizer, 8192, depth, needle, question)

        for cap in caps:
            gc.collect(); torch.cuda.empty_cache()
            cache = middleware.create_cache(device)
            t0 = time.perf_counter()
            last_logits = middleware.chunked_prefill(
                prompt.input_ids.to(device), cache,
                chunk_size=CHUNK_SIZE, prefill_hot_cap=cap,
            )
            ptime = time.perf_counter() - t0
            logits = last_logits.float().cpu()

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

            kl_str = ""
            if cap is None:
                needle_baseline_logits[depth] = logits
            elif depth in needle_baseline_logits:
                bl = F.softmax(needle_baseline_logits[depth][0], dim=-1)
                cl = F.softmax(logits[0], dim=-1)
                kl = F.kl_div(cl.log(), bl, reduction="sum").item()
                kl_str = f"{kl:.4f}"

            print(
                f"  {depth:>5.0%} | {cap_labels[cap]:>12} | "
                f"{'PASS' if hit else 'FAIL':>4} | {ptime:>7.1f}s | {kl_str:>8} | {safe_str(resp, 50)}",
                flush=True,
            )
        print(flush=True)

    middleware.uninstall()
    print(f"\nFinished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
