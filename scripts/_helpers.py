"""Shared helpers for benchmark scripts."""

import io
import os
import random
import sys
import time

# Fix Windows console encoding before any output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Project root on import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from kiv import KIVConfig, KIVMiddleware


def load_model(model_id: str = "google/gemma-4-E2B-it", attn_implementation: str = "sdpa"):
    """
    Load a model in 4-bit NF4 quantization.

    Returns (model, tokenizer, device).
    """
    print(f"Loading {model_id}...", flush=True)
    t0 = time.perf_counter()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    device = next(model.parameters()).device

    print(f"Loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    return model, tokenizer, device


def generate_with_kiv(
    model, tokenizer, middleware: KIVMiddleware, text: str,
    max_new_tokens: int = 60, chunk_size: int = 4096,
    prefill_hot_cap: int | None = None,
):
    """
    Generate text using KIV middleware with chunked prefill.

    Returns (response_text, cache).
    """
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": text}]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"].to(device)

    cache = middleware.create_cache(device)
    last_logits = middleware.chunked_prefill(
        input_ids, cache, chunk_size=chunk_size, prefill_hot_cap=prefill_hot_cap,
    )

    generated = []
    for _ in range(max_new_tokens):
        next_token = last_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if next_token.item() == tokenizer.eos_token_id:
            break
        with torch.no_grad():
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        last_logits = outputs.logits[:, -1, :]

    if generated:
        gen_ids = torch.cat(generated, dim=-1)
        resp = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    else:
        resp = ""

    return resp, cache


def safe_str(text: str, max_len: int = 80) -> str:
    """ASCII-safe truncation for console output."""
    return text[:max_len].encode("ascii", errors="replace").decode("ascii")


_FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony",
    "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly",
    "Andrew", "Emily", "Paul", "Donna", "Joshua", "Michelle", "Kenneth",
    "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca",
    "Jason", "Sharon", "Jeffrey", "Laura", "Ryan", "Cynthia",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Evans", "Turner", "Phillips", "Parker", "Edwards",
    "Collins", "Stewart", "Morris", "Murphy", "Cook",
]

_CITIES = [
    "Springfield", "Portland", "Franklin", "Clinton", "Georgetown",
    "Arlington", "Salem", "Madison", "Chester", "Fairview",
    "Riverside", "Oakland", "Burlington", "Greenville", "Bristol",
    "Manchester", "Dayton", "Dover", "Milton", "Oxford",
]


def make_phonebook(num_entries: int, seed: int = 42) -> list[dict]:
    """Generate deterministic phonebook entries with unique names."""
    rng = random.Random(seed)
    entries = []
    used = set()
    for _ in range(num_entries):
        while True:
            fn = rng.choice(_FIRST_NAMES)
            ln = rng.choice(_LAST_NAMES)
            name = f"{fn} {ln}"
            if name not in used:
                used.add(name)
                break
        phone = f"{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
        city = rng.choice(_CITIES)
        age = rng.randint(22, 78)
        entries.append({"name": name, "phone": phone, "city": city, "age": age})
    return entries


def format_phonebook(entries: list[dict]) -> str:
    """Format phonebook entries as dense text."""
    return "\n".join(
        f"{e['name']}, Phone: {e['phone']}, City: {e['city']}, Age: {e['age']}"
        for e in entries
    )
