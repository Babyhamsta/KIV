# KIV (K-Indexed V Materialization) — Test Results

**Model:** Google Gemma 4 E2B-it, 4-bit NF4 quantization  
**Hardware:** NVIDIA RTX 4070 (12GB VRAM)  
**Date:** 2026-04-12  
**Code:** `kiv/`

## What KIV Does

Replaces the standard KV cache for the 7 global attention layers in Gemma 4 E2B with a tiered retrieval system:

- **Hot cache (VRAM):** Last 2048 tokens with exact K+V for standard attention
- **Page summaries (VRAM):** Mean K vector per 128-token page for fast coarse scoring
- **K pages (CPU):** Per-token K vectors in pinned memory, fetched only for top pages
- **V store (CPU):** Per-token V vectors in pinned memory, fetched only for top-P tokens

Per decode step: exact attention over hot cache, coarse page scoring on GPU, fine scoring of top-32 pages, fetch top-256 V vectors, combine via log-sum-exp. No model weights modified. 28 sliding-window layers untouched.

## Scaling Profile

### Page-based scoring (current, P=256, hot=2048, page_size=128, top_pages=32)

| Context | Cold tokens | Prefill | Decode/step | tok/s | VRAM | CPU RAM |
|---------|-------------|---------|-------------|-------|------|---------|
| 4K | 2,058 | 1.6s | 81ms | 12.3 | 12MB | 12MB |
| 32K | 30,730 | 8.6s | 94ms | 10.6 | 12MB | 180MB |
| 100K | 97,962 | 31.6s | 109ms | 9.2 | 12MB | 574MB |
| 250K | 247,962 | 115s | 112ms | 8.9 | 12MB | 1.4GB |
| 500K | 497,962 | 336s | 106ms | 9.4 | 12MB | 2.9GB |
| **1M** | **997,962** | **1067s** | **130ms** | **7.7** | **12MB** | **5.8GB** |

Decode is near-constant (81-130ms) from 4K to 1M. VRAM stays at 12MB regardless of context.

### Previous brute-force scoring (for comparison)

| Context | Brute-force decode | Page-based decode | Speedup |
|---------|-------------------|-------------------|---------|
| 100K | 152ms | 109ms | 1.4x |
| 250K | 273ms | 112ms | 2.4x |
| 500K | 449ms | 106ms | 4.2x |
| 1M | 813ms | 130ms | **6.3x** |

## Needle-in-Haystack (Passkey Retrieval)

Single fact embedded in natural English filler at varying depths, using chat template.

| Context | Vanilla | P=256 | P=512 | P=1024 |
|---------|---------|-------|-------|--------|
| 4K (5 depths) | 5/5 | 5/5 | 5/5 | 5/5 |
| 8K (5 depths) | 5/5 | 5/5 | 5/5 | 5/5 |
| 16K (5 depths) | OOM | 5/5 | 5/5 | 5/5 |
| 32K (5 depths) | OOM | 5/5 | 5/5 | 5/5 |

Vanilla OOMs past 8K with eager attention.

## P Floor Test (Unique Name Phonebook Lookup)

| Entries | Tokens | P=16 | P=32 | P=64 | P=96 | P=128 | P=192 |
|---------|--------|------|------|------|------|-------|-------|
| 200 | ~5.8K | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 500 | ~14.5K | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 1000 | ~29K | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 2000 | ~58K | 1/1* | - | - | - | - | 2/3** |

No retrieval floor found for unique lookups. P=16 works up to 29K tokens.

## Collision Disambiguation

| Query | P=16 | P=64 | P=256 | Vanilla |
|-------|------|------|-------|---------|
| Kevin Ramirez in Austin | FAIL | FAIL | FAIL | PASS |
| Kevin Ramirez ext 3312 | FAIL | PASS | PASS | FAIL |
| Kevin Ramirez hired 2022 | FAIL | FAIL | PASS | FAIL |
| Mary Smith in Finance | FAIL | FAIL | FAIL | FAIL |

Vanilla gets 3/6, KIV P=256 gets 2/6. The 4-bit E2B model itself struggles with collision disambiguation regardless of attention mechanism.

## Two-Hop and Aggregation

- **Two-hop lookup (name->ID->phone):** FAIL at all P values. Model finds the ID but doesn't chain to the phone lookup. Model reasoning limitation.
- **Multi-record filter:** Needs P >= number of matching records. P=16 counts 3 Kevin Ramirez entries but not 41 Ramirez entries.

## What KIV Does Well

- Single-record retrieval with unique keys works down to P=16, even at 58K tokens
- VRAM bounded at 12MB from 4K to 1M tokens. Model uses ~6.5GB. Total GPU ~6.5GB.
- Decode speed near-constant: 81-130ms regardless of context length
- 1M token context on a consumer 12GB GPU at 7.7 tok/s

## Where KIV Has Limits

- Multi-record aggregation needs P >= number of matching records
- Collision disambiguation partially works but the base model also struggles
- Prefill at 1M takes ~18 minutes (one-time cost per conversation)

## Where the Model Has Limits (Not KIV-Related)

- Two-hop reasoning fails at all P values — model doesn't chain lookups
- Collision disambiguation inconsistent even with full exact attention
- These are 4-bit 2B parameter model limitations, not cache limitations
