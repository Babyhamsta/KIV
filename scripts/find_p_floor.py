"""Find the minimum P that still passes phonebook lookup."""
import gc
import time

from _helpers import (
    KIVConfig, KIVMiddleware,
    generate_with_kiv, load_model, make_phonebook, format_phonebook, safe_str,
)


def main():
    import torch

    print("=== KIV Floor Finder ===")
    print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.perf_counter()

    model, tokenizer, device = load_model(attn_implementation="eager")

    p_values = [192, 128, 96, 64, 32, 16]
    entry_counts = [200, 500, 1000, 2000]
    results = {}  # (P, num_entries, target_name) -> bool

    for num_entries in entry_counts:
        entries = make_phonebook(num_entries)
        phonebook_text = format_phonebook(entries)
        tok_count = len(tokenizer.encode(phonebook_text, add_special_tokens=False))

        targets = [
            entries[num_entries // 4],
            entries[num_entries // 2],
            entries[3 * num_entries // 4],
        ]

        print(f"{'=' * 80}", flush=True)
        print(f"Phonebook: {num_entries} entries (~{tok_count} tokens)", flush=True)
        print(f"{'=' * 80}", flush=True)

        all_failing = True

        for top_p in p_values:
            passes = 0
            config = KIVConfig(hot_budget=2048, top_p=top_p)
            middleware = KIVMiddleware(model, config)
            middleware.install()

            for target in targets:
                query = f"What is {target['name']}'s phone number?"
                full_text = f"Here is a phone directory:\n\n{phonebook_text}\n\n{query}"

                gc.collect(); torch.cuda.empty_cache()
                t_start = time.perf_counter()
                resp, cache = generate_with_kiv(model, tokenizer, middleware, full_text)
                elapsed = time.perf_counter() - t_start
                hit = target["phone"] in resp

                results[(top_p, num_entries, target["name"])] = hit
                if hit:
                    passes += 1

                print(
                    f"  [P={top_p:4d}] {target['name']:20s}: {'PASS' if hit else 'FAIL'} ({elapsed:.1f}s) | {safe_str(resp, 70)}",
                    flush=True,
                )

            middleware.uninstall()

            if passes > 0:
                all_failing = False

            print(f"  P={top_p}: {passes}/3 passed\n", flush=True)

        if all_failing:
            print(f"  ALL P values failed at {num_entries} entries. Stopping.", flush=True)
            break

    # Summary grid
    total_elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 80}", flush=True)
    print(f"SUMMARY (total time: {total_elapsed:.0f}s)", flush=True)
    print(f"{'=' * 80}", flush=True)

    tested_entries = sorted(set(ne for (_, ne, _) in results))
    tested_p = sorted(set(p for (p, _, _) in results), reverse=True)

    print(f"\n{'P':>6} |", end="")
    for ne in tested_entries:
        print(f" {ne:>5} entries |", end="")
    print()
    print("-" * (8 + 15 * len(tested_entries)))

    for p in tested_p:
        row = f"{p:>6} |"
        for ne in tested_entries:
            hits = sum(1 for (pp, nn, _), v in results.items() if pp == p and nn == ne and v)
            total = sum(1 for (pp, nn, _) in results if pp == p and nn == ne)
            if total == 0:
                row += "       -     |"
            else:
                row += f"     {hits}/{total}     |"
        print(row)

    print(f"\nFinished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
