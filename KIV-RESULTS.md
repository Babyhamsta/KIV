# KIV (K-Indexed V Materialization) — Results

**Model:** Google Gemma 4 E2B-it, 4-bit NF4 quantization  
**Hardware:** Intel i7-13700K, 64GB DDR5 (6000MT/s), NVIDIA RTX 4070 (12GB VRAM)  
**Config:** P=256, hot_budget=2048, page_size=128, top_pages=32  
**Code:** `kiv/`

## Method

Tiered KV cache on the 7 global attention layers. 28 sliding-window layers untouched, model weights unmodified.

- **Hot (VRAM):** Last 2048 tokens, exact K+V
- **Page summaries (VRAM):** Mean K per 128-token page for coarse scoring
- **K pages (CPU):** Per-token K, fetched for top-32 pages
- **V store (CPU):** Per-token V, fetched for top-256 tokens

Per decode step: coarse page scoring → fine K scoring → fetch top-P K/V from CPU → concatenate with hot → attention over 2304 tokens.

## Scaling Profile

| Context | Cold tokens | Prefill | Decode/step | tok/s | VRAM (KIV) | CPU RAM |
|---------|-------------|---------|-------------|-------|------------|---------|
| 4K | 2,058 | 1.5s | 77ms | 12.9 | 12MB | 12MB |
| 32K | 30,730 | 7.8s | 110ms | 9.1 | 12MB | 180MB |
| 100K | 97,962 | 23.4s | 122ms | 8.2 | 12MB | 574MB |
| 250K | 247,962 | 58.4s | 142ms | 7.0 | 12MB | 1.4GB |
| 500K | 497,962 | 128.9s | 182ms | 5.5 | 12MB | 2.9GB |
| 1M | 997,962 | 258.8s | 243ms | 4.1 | 12MB | 5.8GB |

KIV VRAM stays at 12MB regardless of context length. Model weights use ~6.5GB. CPU RAM scales linearly with cold token count.

## Needle-in-Haystack (Passkey Retrieval)

Single fact embedded in natural English filler at varying depths, using chat template.

| Context | Vanilla | P=256 | P=512 | P=1024 |
|---------|---------|-------|-------|--------|
| 4K (5 depths) | 5/5 | 5/5 | 5/5 | 5/5 |
| 8K (5 depths) | 5/5 | 5/5 | 5/5 | 5/5 |
| 16K (5 depths) | OOM | 5/5 | 5/5 | 5/5 |
| 32K (5 depths) | OOM | 5/5 | 5/5 | 5/5 |

70/70 tests passed. Vanilla OOMs past 8K with eager attention on this hardware.

## Phonebook Lookup (Unique Names)

| Entries | Tokens | P=16 | P=64 | P=256 |
|---------|--------|------|------|-------|
| 200 | ~5.8K | 3/3 | 3/3 | 3/3 |
| 500 | ~14.5K | 3/3 | 3/3 | 3/3 |
| 1000 | ~29K | 1/3 | 2/3 | 3/3 |

P=16 and P=64 retrieve correctly up to 14.5K tokens. At 29K tokens, P=256 is needed for reliable retrieval across all target positions.

## Collision Disambiguation

| Query | P=16 | P=64 | P=256 |
|-------|------|------|-------|
| Kevin Ramirez in Austin | FAIL | FAIL | FAIL |
| Kevin Ramirez ext 3312 | FAIL | FAIL | FAIL |
| Kevin Ramirez hired 2022 | FAIL | FAIL | FAIL |
| Mary Smith in Finance | FAIL | FAIL | FAIL |

0/12 across all P values. The 4-bit E2B model cannot disambiguate duplicate names by secondary attributes even with exact attention (vanilla also fails this).

## Multi-Needle (10 Facts Scattered in Context)

| Fact | P=16 | P=64 | P=256 |
|------|------|------|-------|
| Q3 budget ($847,293) | PASS | PASS | PASS |
| CEO birthday (March 14) | PASS | PASS | PASS |
| Fire alarm code (7742) | PASS | PASS | PASS |

9/9 across all P values. Distinctive facts are reliably retrieved regardless of P.

## Two-Hop and Aggregation

- **Two-hop lookup (name->ID->phone):** FAIL at all P values. Model finds the ID but does not chain to the phone lookup (vanilla also fails).
- **Distant context reasoning:** PASS at all P values. Two premises separated by thousands of tokens are correctly combined (employee ID 4821 -> 7th floor).
- **Multi-record filter:** Kevin Ramirez count (4 entries): 3/3 PASS. Full Ramirez count (47 entries): 0/3 FAIL — requires P >= matching record count.

## Summary

Single-record lookup works at P=64 up to 14.5K tokens, P=256 up to 29K. Multi-needle with distinctive facts works at P=16. VRAM fixed at 12MB from 4K to 1M. Prefill at 1M takes ~4.3 minutes.

Multi-record aggregation needs P >= matching record count. Collision disambiguation and two-hop reasoning fail regardless of P — vanilla model also fails these (4-bit 2B model constraint).
