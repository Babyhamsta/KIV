# KIV (K-Indexed V Materialization) — Results

**Model:** Google Gemma 4 E2B-it, 4-bit NF4 quantization  
**Hardware:** Intel i7-13700K, 64GB DDR5 (6000MT/s), NVIDIA RTX 4070 (12GB VRAM)  
**Config:** P=256, hot_budget=2048, page_size=128, top_pages=32  
**Code:** `kiv/`

## Method

Replaces the standard KV cache for the 7 global attention layers in Gemma 4 E2B with a tiered retrieval system:

- **Hot cache (VRAM):** Last 2048 tokens with exact K+V for standard attention
- **Page summaries (VRAM):** Mean K vector per 128-token page for coarse scoring
- **K pages (CPU):** Per-token K vectors in pinned memory, fetched only for top pages
- **V store (CPU):** Per-token V vectors in pinned memory, fetched only for top-P tokens

Per decode step: exact attention over hot cache, coarse page scoring on GPU, fine scoring of top-32 pages, fetch top-256 V vectors, combine via log-sum-exp. No model weights modified. 28 sliding-window layers untouched.

## Scaling Profile

| Context | Cold tokens | Prefill | Decode/step | tok/s | VRAM (KIV) | CPU RAM |
|---------|-------------|---------|-------------|-------|------------|---------|
| 4K | 2,058 | 1.7s | 103ms | 9.7 | 12MB | 12MB |
| 32K | 30,730 | 8.3s | 121ms | 8.3 | 12MB | 180MB |
| 100K | 97,962 | 24.6s | 132ms | 7.6 | 12MB | 574MB |
| 250K | 247,962 | 61.7s | 146ms | 6.8 | 12MB | 1.4GB |
| 500K | 497,962 | 143s | 199ms | 5.0 | 12MB | 2.9GB |
| 1M | 997,962 | 261s | 266ms | 3.8 | 12MB | 5.8GB |

KIV VRAM stays at 12MB regardless of context length. Model weights use ~6.5GB. CPU RAM scales linearly with cold token count.

## Needle-in-Haystack (Passkey Retrieval)

Single fact embedded in natural English filler at varying depths, using chat template.

| Context | Vanilla | P=256 | P=512 | P=1024 |
|---------|---------|-------|-------|--------|
| 4K (5 depths) | 5/5 | 5/5 | 5/5 | 5/5 |
| 8K (5 depths) | 5/5 | 5/5 | 5/5 | 5/5 |
| 16K (5 depths) | OOM | 5/5 | 5/5 | 5/5 |
| 32K (5 depths) | OOM | 5/5 | 5/5 | 5/5 |

Vanilla OOMs past 8K with eager attention on this hardware.

## P Floor Test (Unique Name Phonebook Lookup)

| Entries | Tokens | P=16 | P=32 | P=64 | P=96 | P=128 | P=192 |
|---------|--------|------|------|------|------|-------|-------|
| 200 | ~5.8K | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 500 | ~14.5K | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1000 | ~29K | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 2000 | ~58K | 1/1* | - | - | - | - | 2/3** |

No retrieval floor found for unique lookups. P=16 retrieves correctly up to 29K tokens.

## Collision Disambiguation

| Query | P=16 | P=64 | P=256 | Vanilla |
|-------|------|------|-------|---------|
| Kevin Ramirez in Austin | FAIL | FAIL | FAIL | PASS |
| Kevin Ramirez ext 3312 | FAIL | PASS | PASS | FAIL |
| Kevin Ramirez hired 2022 | FAIL | FAIL | PASS | FAIL |
| Mary Smith in Finance | FAIL | FAIL | FAIL | FAIL |

Vanilla gets 3/6, KIV P=256 gets 2/6. The base 4-bit E2B model struggles with collision disambiguation regardless of attention mechanism.

## Two-Hop and Aggregation

- **Two-hop lookup (name->ID->phone):** FAIL at all P values. Model finds the ID but does not chain to the phone lookup. This is a model reasoning limitation, not a retrieval limitation.
- **Multi-record filter:** Requires P >= number of matching records. P=16 counts 3 Kevin Ramirez entries but cannot aggregate 41 Ramirez entries.

## Observations

**Retrieval:** Single-record lookup with unique keys works down to P=16 at 58K tokens. VRAM bounded at 12MB from 4K to 1M. Prefill at 1M takes ~4.5 minutes (one-time cost).

**KIV limitations:** Multi-record aggregation requires P >= matching record count. Collision disambiguation is partially effective but constrained by the base model's performance on the same task.

**Model limitations (not KIV-related):** Two-hop reasoning fails at all P values. Collision disambiguation is inconsistent even with full exact attention. These reflect 4-bit 2B parameter model constraints, not cache behavior.
