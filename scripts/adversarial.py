"""
Adversarial retrieval tests for KIV.

Tests designed to stress the top-P retrieval mechanism beyond simple passkey lookup:
1. Dense factual (phonebook) - similar entries in K-space
2. Collision disambiguation - duplicate names, disambiguate by attribute
3. Multi-needle - 10 facts scattered in context, ask about specific one
4. Paraphrase retrieval - different wording between stored text and query
5. Two-hop lookup - name->ID->phone requires chaining two records
6. Reasoning over distant context - combine premises thousands of tokens apart
7. Multi-record filter - aggregate queries over multiple matching entries
"""
import gc
import random
import time

from _helpers import (
    KIVConfig, KIVMiddleware,
    generate_with_kiv, load_model, make_phonebook, format_phonebook, safe_str,
)


def main():
    print("=== KIV Adversarial Tests ===")
    print(f"Start: {time.strftime('%H:%M:%S')}", flush=True)

    model, tokenizer, device = load_model(attn_implementation="eager")

    import torch
    from kiv.eval_utils import FILLER_PARAGRAPHS

    p_values = [16, 64, 256]

    def run_test(middleware, label, text, check_fn, max_new_tokens=60):
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.perf_counter()
        resp, cache = generate_with_kiv(model, tokenizer, middleware, text, max_new_tokens)
        elapsed = time.perf_counter() - t0
        hit = check_fn(resp)
        mem = cache.memory_report()
        vram_mb = mem["total_vram_bytes"] / 1024 / 1024
        label = safe_str(label, 40)
        print(f"    [{label}] {'PASS' if hit else 'FAIL'} ({elapsed:.1f}s, {vram_mb:.0f}MB) | {safe_str(resp)}", flush=True)
        return hit

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Dense Factual (Phonebook)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("TEST 1: Dense Factual (Phonebook Lookup)", flush=True)
    print("=" * 90, flush=True)

    for num_entries in [200, 500, 1000]:
        entries = make_phonebook(num_entries)
        phonebook = format_phonebook(entries)
        tok_count = len(tokenizer.encode(phonebook, add_special_tokens=False))
        targets = [entries[num_entries // 4], entries[num_entries // 2], entries[3 * num_entries // 4]]
        print(f"\n  {num_entries} entries (~{tok_count} tokens)", flush=True)

        for top_p in p_values:
            config = KIVConfig(hot_budget=2048, top_p=top_p)
            mw = KIVMiddleware(model, config); mw.install()
            for t in targets:
                query = f"What is {t['name']}'s phone number?"
                run_test(mw, f"P={top_p} {t['name']}", f"Here is a phone directory:\n\n{phonebook}\n\n{query}",
                         lambda r, exp=t['phone']: exp in r)
            mw.uninstall()

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Collision Disambiguation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("TEST 2: Collision Disambiguation", flush=True)
    print("=" * 90, flush=True)

    rng = random.Random(42)
    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
                   "Linda", "David", "Elizabeth", "William", "Barbara", "Kevin", "Susan"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Ramirez",
                  "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas", "Taylor"]
    cities = ["Austin", "Portland", "Seattle", "Denver", "Chicago", "Boston", "Miami",
              "Phoenix", "Dallas", "Atlanta"]
    depts = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Legal", "Support"]

    collisions = [
        {"name": "Kevin Ramirez", "phone": "512-334-7721", "city": "Austin", "dept": "Engineering", "ext": "4421", "hired": "2019"},
        {"name": "Kevin Ramirez", "phone": "303-887-2290", "city": "Denver", "dept": "Marketing", "ext": "5508", "hired": "2022"},
        {"name": "Kevin Ramirez", "phone": "617-442-1893", "city": "Boston", "dept": "Sales", "ext": "3312", "hired": "2017"},
        {"name": "Mary Smith", "phone": "480-221-9934", "city": "Phoenix", "dept": "HR", "ext": "6677", "hired": "2020"},
        {"name": "Mary Smith", "phone": "214-553-8812", "city": "Dallas", "dept": "Finance", "ext": "2245", "hired": "2018"},
    ]
    col_entries = list(collisions)
    used = set()
    for _ in range(495):
        while True:
            fn, ln, city = rng.choice(first_names), rng.choice(last_names), rng.choice(cities)
            combo = f"{fn}_{ln}_{city}"
            if combo not in used: used.add(combo); break
        col_entries.append({"name": f"{fn} {ln}", "phone": f"{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}",
                            "city": city, "dept": rng.choice(depts), "ext": str(rng.randint(1000,9999)), "hired": str(rng.randint(2010,2024))})
    rng.shuffle(col_entries)
    col_phonebook = "\n".join(f"{e['name']}, Phone: {e['phone']}, City: {e['city']}, Dept: {e['dept']}, Ext: {e['ext']}, Hired: {e['hired']}" for e in col_entries)

    col_tests = [
        ("Kevin Ramirez in Austin?", "512-334-7721"),
        ("Kevin Ramirez ext 3312?", "617-442-1893"),
        ("Kevin Ramirez hired 2022?", "303-887-2290"),
        ("Mary Smith in Finance?", "214-553-8812"),
    ]

    for top_p in p_values:
        config = KIVConfig(hot_budget=2048, top_p=top_p)
        mw = KIVMiddleware(model, config); mw.install()
        print(f"\n  P={top_p}:", flush=True)
        for q, exp in col_tests:
            run_test(mw, q, f"Here is an employee directory:\n\n{col_phonebook}\n\nWhat is the phone number for {q}", lambda r, e=exp: e in r)
        mw.uninstall()

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Multi-Needle
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("TEST 3: Multi-Needle (10 facts, ask about specific one)", flush=True)
    print("=" * 90, flush=True)

    facts = [
        ("The project budget for Q3 is exactly $847,293.", "budget", "$847,293"),
        ("The server room temperature must be kept at 18.5 degrees Celsius.", "server room temperature", "18.5"),
        ("The emergency evacuation meeting point is at Parking Lot C.", "evacuation meeting point", "Parking Lot C"),
        ("The WiFi password for the guest network is 'sunflower2024'.", "WiFi password", "sunflower2024"),
        ("The CEO's birthday is on March 14th.", "CEO's birthday", "March 14"),
        ("The company was founded in 1987 in Portland, Oregon.", "company founded", "1987"),
        ("The maximum allowed file upload size is 250 megabytes.", "max file upload size", "250"),
        ("The next board meeting is scheduled for November 8th at 2pm.", "next board meeting", "November 8"),
        ("The building's fire alarm code is 7742.", "fire alarm code", "7742"),
        ("The annual revenue target for this year is $12.4 million.", "annual revenue target", "$12.4 million"),
    ]
    filler_pool = FILLER_PARAGRAPHS * 5
    paragraphs = list(filler_pool)
    for i, (fact, _, _) in enumerate(facts):
        pos = int(len(paragraphs) * (i + 0.5) / len(facts))
        paragraphs.insert(pos, fact)
    doc = "\n\n".join(paragraphs)

    test_facts = [facts[0], facts[4], facts[8]]
    for top_p in p_values:
        config = KIVConfig(hot_budget=2048, top_p=top_p)
        mw = KIVMiddleware(model, config); mw.install()
        print(f"\n  P={top_p}:", flush=True)
        for _, topic, expected in test_facts:
            q = f"Based on the document above, what is the {topic}?"
            run_test(mw, topic, f"Read this document carefully:\n\n{doc}\n\n{q}", lambda r, e=expected: e.lower() in r.lower())
        mw.uninstall()

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Two-Hop Lookup
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("TEST 4: Two-Hop Lookup (name -> ID -> phone)", flush=True)
    print("=" * 90, flush=True)

    rng2 = random.Random(99)
    dir_a = []
    for i in range(300):
        fn = first_names[i % len(first_names)]
        ln = last_names[i // len(first_names) % len(last_names)]
        name = f"{fn} {ln}" if i < len(first_names) * len(last_names) else f"{fn} {ln} {i}"
        dir_a.append({"name": name, "emp_id": f"EMP-{rng2.randint(10000,99999)}"})
    target_a = dir_a[150]
    dir_b = [{"emp_id": e["emp_id"], "phone": f"{rng2.randint(200,999)}-{rng2.randint(100,999)}-{rng2.randint(1000,9999)}"} for e in dir_a]
    target_b = next(e for e in dir_b if e["emp_id"] == target_a["emp_id"])
    rng2.shuffle(dir_a); rng2.shuffle(dir_b)
    doc_2hop = "EMPLOYEE DIRECTORY (Name -> ID):\n" + "\n".join(f"{e['name']}: {e['emp_id']}" for e in dir_a) + \
               "\n\n---\n\nPHONE DIRECTORY (ID -> Phone):\n" + "\n".join(f"{e['emp_id']}: {e['phone']}" for e in dir_b)

    for top_p in p_values:
        config = KIVConfig(hot_budget=2048, top_p=top_p)
        mw = KIVMiddleware(model, config); mw.install()
        q = f"What is {target_a['name']}'s phone number? Look up their employee ID first, then find the phone number for that ID."
        run_test(mw, f"P={top_p} two-hop", f"{doc_2hop}\n\n{q}", lambda r: target_b["phone"] in r)
        mw.uninstall()

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Reasoning Over Distant Context
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("TEST 5: Reasoning Over Distant Context", flush=True)
    print("=" * 90, flush=True)

    premise_a = "Alice works in the marketing department and her employee ID number is 4821."
    premise_b = "The marketing department is located on the 7th floor of the Westbrook building."
    filler_block = "\n\n".join(FILLER_PARAGRAPHS * 4)
    third = len(filler_block) // 3
    dist_doc = filler_block[:third] + f"\n\n{premise_a}\n\n" + filler_block[third:2*third] + f"\n\n{premise_b}\n\n" + filler_block[2*third:]

    for top_p in p_values:
        config = KIVConfig(hot_budget=2048, top_p=top_p)
        mw = KIVMiddleware(model, config); mw.install()
        run_test(mw, f"P={top_p} distant", f"Read this document carefully:\n\n{dist_doc}\n\nOn which floor does the employee with ID 4821 work?",
                 lambda r: "7" in r)
        mw.uninstall()

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Multi-Record Filter
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("TEST 6: Multi-Record Filter", flush=True)
    print("=" * 90, flush=True)

    ramirez_count = sum(1 for e in col_entries if "Ramirez" in e["name"])
    kevin_ramirez_count = sum(1 for e in col_entries if e["name"] == "Kevin Ramirez")

    for top_p in p_values:
        config = KIVConfig(hot_budget=2048, top_p=top_p)
        mw = KIVMiddleware(model, config); mw.install()
        print(f"\n  P={top_p}:", flush=True)
        run_test(mw, f"Count Kevin Ramirez (expect {kevin_ramirez_count})",
                 f"Here is an employee directory:\n\n{col_phonebook}\n\nHow many Kevin Ramirez entries are there, and what are their different cities?",
                 lambda r: (str(kevin_ramirez_count) in r or "three" in r.lower()))
        run_test(mw, f"Count Ramirez (expect {ramirez_count})",
                 f"Here is an employee directory:\n\n{col_phonebook}\n\nHow many people named Ramirez are in the directory?",
                 lambda r: str(ramirez_count) in r)
        mw.uninstall()

    total = time.perf_counter()
    print(f"\n{'=' * 90}", flush=True)
    print(f"All adversarial tests complete. Finished at {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
