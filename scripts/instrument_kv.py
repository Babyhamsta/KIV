"""
Instrument Gemma 4 E2B original model to dump per-layer KV statistics
for global (full) attention layers.

Captures:
  1. Raw K, V tensors (after projection + norm + RoPE, before attention)
  2. SVD spectra (singular values, cumulative explained variance)
  3. Per-head cosine similarity (K redundancy across heads)
  4. Temporal deltas (||K_t - K_{t-1}|| and ||V_t - V_{t-1}||)
  5. Attention weight statistics (mean/max per token)
  6. Low-rank reconstruction error (rank 32/64/128)

Usage:
    python scripts/instrument_kv.py --seq-len 8192
    python scripts/instrument_kv.py --seq-len 8192 --output-dir kv_analysis
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="KV instrumentation for Gemma 4 E2B")
    p.add_argument("--model-id", default="google/gemma-4-E2B-it")
    p.add_argument("--seq-len", type=int, default=8192, help="Sequence length to analyze")
    p.add_argument("--output-dir", default="kv_analysis", help="Output directory")
    p.add_argument("--save-raw-kv", action="store_true", help="Save raw K/V tensors (large)")
    p.add_argument("--low-rank-tests", nargs="+", type=int, default=[32, 64, 128],
                   help="Ranks to test for low-rank reconstruction")
    return p.parse_args()


# ── Global attention layer indices in Gemma 4 E2B ──
# 4:1 sliding:full pattern, layers [4, 9, 14, 19, 24, 29, 34]
# Layers 19+ are KV-shared (reuse from layer 14), so only 4, 9, 14
# have independent K/V projections.
FULL_ATTN_LAYERS = [4, 9, 14, 19, 24, 29, 34]
INDEPENDENT_KV_LAYERS = [4, 9, 14]  # layers that actually compute K/V


def monkey_patch_eager_attention(model_module):
    """
    Patch the eager attention forward to capture attention weight STATISTICS
    (not raw weights — those are too large at 8K+ sequences).
    Returns a dict populated with {layer_idx: stats_dict}.
    """
    captured_stats = {}

    import transformers.models.gemma4.modeling_gemma4 as gemma4_mod
    original_eager = gemma4_mod.eager_attention_forward

    def capturing_eager(module, query, key, value, attention_mask, **kwargs):
        attn_output, attn_weights = original_eager(
            module, query, key, value, attention_mask, **kwargs
        )
        if hasattr(module, "layer_idx") and module.layer_idx in FULL_ATTN_LAYERS:
            if attn_weights is not None:
                # Compute stats on GPU, store only summaries
                w = attn_weights[0].float()  # [H, T, T]
                H, T, _ = w.shape
                # Mean/max attention received per token
                mean_recv = w.mean(dim=0).mean(dim=0)  # [T]
                max_recv = w.max(dim=0).values.max(dim=0).values  # [T]
                # Entropy per head
                log_w = torch.log(w.clamp(min=1e-10))
                entropy = -(w * log_w).sum(dim=-1).mean(dim=-1)  # [H]
                # Sink attention (first 4 tokens)
                sink = w[:, :, :4].sum(dim=-1).mean(dim=-1)  # [H]
                captured_stats[module.layer_idx] = {
                    "mean_attn_received_first_64": mean_recv[:64].cpu().numpy().tolist(),
                    "mean_attn_received_last_64": mean_recv[-64:].cpu().numpy().tolist(),
                    "max_attn_received_first_64": max_recv[:64].cpu().numpy().tolist(),
                    "per_head_entropy": entropy.cpu().numpy().tolist(),
                    "per_head_sink_attention": sink.cpu().numpy().tolist(),
                    "overall_mean_entropy": float(entropy.mean().cpu()),
                }
                del w, mean_recv, max_recv, entropy, sink, log_w
        return attn_output, attn_weights

    gemma4_mod.eager_attention_forward = capturing_eager
    return captured_stats, original_eager, gemma4_mod


def register_kv_hooks(model):
    """
    Register forward hooks on full_attention layers to capture K and V
    after projection, norm, and RoPE (right before attention computation).
    """
    captured_kv = {}
    hooks = []

    text_model = model.model.language_model

    for layer_idx in FULL_ATTN_LAYERS:
        attn = text_model.layers[layer_idx].self_attn

        def make_hook(idx):
            def hook_fn(module, args, kwargs, output):
                # After forward completes, intercept the internal state.
                # We need to re-extract K/V by running projections again.
                # This is cheaper than modifying the forward.
                pass
            return hook_fn

    # Instead of hooks on forward output, we'll hook into the forward
    # to capture K/V mid-computation. Use a wrapper approach.
    original_forwards = {}

    for layer_idx in FULL_ATTN_LAYERS:
        attn = text_model.layers[layer_idx].self_attn
        original_forwards[layer_idx] = attn.forward

        def make_capturing_forward(orig_forward, idx, attn_mod):
            def capturing_forward(hidden_states, position_embeddings, attention_mask,
                                  shared_kv_states, past_key_values=None, **kwargs):
                from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_mod.head_dim)
                cos, sin = position_embeddings

                # Capture K and V after full processing
                if attn_mod.is_kv_shared_layer:
                    key_states, value_states = shared_kv_states[attn_mod.kv_shared_layer_index]
                    key_states = key_states.to(hidden_states.device)
                    value_states = value_states.to(hidden_states.device)
                    captured_kv[idx] = {
                        "k": key_states.detach().cpu().float(),
                        "v": value_states.detach().cpu().float(),
                        "is_shared": True,
                        "shared_from": attn_mod.kv_shared_layer_index,
                    }
                else:
                    # Recompute K/V to capture them
                    k = attn_mod.k_proj(hidden_states).view(hidden_shape)
                    k = attn_mod.k_norm(k)
                    k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
                    k = k.transpose(1, 2)

                    v_proj = attn_mod.v_proj
                    if v_proj is not None:
                        v = v_proj(hidden_states).view(hidden_shape)
                    else:
                        # K=V mode
                        v = attn_mod.k_proj(hidden_states).view(hidden_shape)
                    v = attn_mod.v_norm(v)
                    v = v.transpose(1, 2)

                    captured_kv[idx] = {
                        "k": k.detach().cpu().float(),
                        "v": v.detach().cpu().float(),
                        "is_shared": False,
                    }

                # Call original forward
                return orig_forward(hidden_states, position_embeddings, attention_mask,
                                    shared_kv_states, past_key_values=past_key_values, **kwargs)

            return capturing_forward

        attn.forward = make_capturing_forward(
            original_forwards[layer_idx], layer_idx, attn
        )

    return captured_kv, original_forwards


def restore_forwards(model, original_forwards):
    text_model = model.model.language_model
    for layer_idx, orig in original_forwards.items():
        text_model.layers[layer_idx].self_attn.forward = orig


def compute_svd_spectra(tensor, name=""):
    """
    Compute SVD spectra for a [B, H, T, D] tensor.
    Returns per-head and concatenated-head results.
    """
    B, H, T, D = tensor.shape
    results = {"per_head": {}, "concatenated": {}}

    # Per-head SVD: reshape to [B*H, T, D]
    flat = tensor.reshape(B * H, T, D)

    # For efficiency, only use first batch element
    for h in range(H):
        mat = tensor[0, h]  # [T, D]
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        total_var = (S ** 2).sum().item()
        cumvar = torch.cumsum(S ** 2, dim=0) / total_var
        results["per_head"][h] = {
            "singular_values": S.numpy().tolist(),
            "cumulative_explained_variance": cumvar.numpy().tolist(),
            "rank_for_90pct": int((cumvar >= 0.90).nonzero(as_tuple=True)[0][0].item()) + 1 if (cumvar >= 0.90).any() else D,
            "rank_for_95pct": int((cumvar >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1 if (cumvar >= 0.95).any() else D,
            "rank_for_99pct": int((cumvar >= 0.99).nonzero(as_tuple=True)[0][0].item()) + 1 if (cumvar >= 0.99).any() else D,
        }

    # Concatenated across heads: [T, H*D]
    concat = tensor[0].permute(1, 0, 2).reshape(T, H * D)  # [T, H*D]
    U, S, Vh = torch.linalg.svd(concat, full_matrices=False)
    total_var = (S ** 2).sum().item()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    results["concatenated"] = {
        "singular_values": S.numpy().tolist()[:256],  # cap for JSON size
        "cumulative_explained_variance": cumvar.numpy().tolist()[:256],
        "rank_for_90pct": int((cumvar >= 0.90).nonzero(as_tuple=True)[0][0].item()) + 1 if (cumvar >= 0.90).any() else H * D,
        "rank_for_95pct": int((cumvar >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1 if (cumvar >= 0.95).any() else H * D,
        "rank_for_99pct": int((cumvar >= 0.99).nonzero(as_tuple=True)[0][0].item()) + 1 if (cumvar >= 0.99).any() else H * D,
    }

    return results


def compute_head_cosine_similarity(tensor):
    """
    Compute pairwise cosine similarity between heads' K vectors.
    tensor: [B, H, T, D] → compute similarity of mean K vector per head.
    """
    B, H, T, D = tensor.shape
    # Mean K vector per head: [H, D]
    head_means = tensor[0].mean(dim=1)  # [H, D]

    # Normalize
    norms = head_means.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = head_means / norms

    # Cosine similarity matrix: [H, H]
    cos_sim = torch.mm(normalized, normalized.t())

    return {
        "cosine_similarity_matrix": cos_sim.numpy().tolist(),
        "mean_off_diagonal": float(cos_sim.fill_diagonal_(0).sum() / (H * (H - 1))),
        "max_off_diagonal": float(cos_sim.max()),
    }


def compute_temporal_deltas(tensor, name=""):
    """
    Compute ||X_t - X_{t-1}|| for each token position.
    tensor: [B, H, T, D]
    """
    B, H, T, D = tensor.shape

    # Deltas: [B, H, T-1, D]
    deltas = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    delta_norms = deltas.norm(dim=-1)  # [B, H, T-1]

    # Per-head stats
    per_head_mean = delta_norms[0].mean(dim=1)  # [H]
    per_head_std = delta_norms[0].std(dim=1)  # [H]

    # Overall
    all_norms = delta_norms[0].reshape(-1)

    return {
        "per_head_mean_delta": per_head_mean.numpy().tolist(),
        "per_head_std_delta": per_head_std.numpy().tolist(),
        "overall_mean_delta": float(all_norms.mean()),
        "overall_std_delta": float(all_norms.std()),
        "overall_median_delta": float(all_norms.median()),
        "overall_max_delta": float(all_norms.max()),
        "percentile_95": float(all_norms.quantile(0.95)),
    }


def compute_low_rank_reconstruction(tensor, ranks):
    """
    Test low-rank reconstruction quality at specified ranks.
    tensor: [B, H, T, D]
    """
    B, H, T, D = tensor.shape
    results = {}

    # Use first batch, concatenated across heads
    concat = tensor[0].permute(1, 0, 2).reshape(T, H * D)  # [T, H*D]
    U, S, Vh = torch.linalg.svd(concat, full_matrices=False)

    original_norm = concat.norm().item()

    for rank in ranks:
        if rank >= min(T, H * D):
            results[rank] = {
                "reconstruction_error": 0.0,
                "relative_error": 0.0,
                "explained_variance": 1.0,
            }
            continue

        # Low-rank approximation
        reconstructed = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
        error = (concat - reconstructed).norm().item()
        rel_error = error / original_norm

        total_var = (S ** 2).sum().item()
        explained = (S[:rank] ** 2).sum().item() / total_var

        results[rank] = {
            "reconstruction_error": error,
            "relative_error": rel_error,
            "explained_variance": explained,
        }

    return results


def compute_attention_stats(attn_weights):
    """
    Compute token importance from attention weights.
    attn_weights: [B, H, T, T] (may be very large)
    """
    B, H, T, _ = attn_weights.shape

    # Mean attention received per token (averaged over heads and query positions)
    # attn_weights[b, h, q, k] = how much query q attends to key k
    mean_attn_received = attn_weights[0].mean(dim=0).mean(dim=0)  # [T] avg over heads and queries
    max_attn_received = attn_weights[0].max(dim=0).values.max(dim=0).values  # [T] max over heads and queries

    # Attention entropy per head (measure of how spread out attention is)
    # Clamp for numerical stability
    log_attn = torch.log(attn_weights[0].clamp(min=1e-10))
    entropy = -(attn_weights[0] * log_attn).sum(dim=-1).mean(dim=-1)  # [H]

    # Attention sink: how much attention goes to first few tokens
    sink_attn = attn_weights[0, :, :, :4].sum(dim=-1).mean(dim=-1)  # [H] mean over queries

    return {
        "mean_attn_received_first_64": mean_attn_received[:64].numpy().tolist(),
        "mean_attn_received_last_64": mean_attn_received[-64:].numpy().tolist(),
        "max_attn_received_first_64": max_attn_received[:64].numpy().tolist(),
        "per_head_entropy": entropy.numpy().tolist(),
        "per_head_sink_attention": sink_attn.numpy().tolist(),
        "overall_mean_entropy": float(entropy.mean()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== KV Instrumentation: seq_len={args.seq_len} ===")

    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention weight capture
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.eval()

    # Print model config for reference
    text_config = model.config.text_config
    print(f"  hidden_size={text_config.hidden_size}")
    print(f"  num_attention_heads={text_config.num_attention_heads}")
    print(f"  num_key_value_heads={text_config.num_key_value_heads}")
    print(f"  global_head_dim={text_config.global_head_dim}")
    print(f"  head_dim={text_config.head_dim}")
    print(f"  num_hidden_layers={text_config.num_hidden_layers}")
    print(f"  sliding_window={text_config.sliding_window}")

    # Check which layers are shared
    for idx in FULL_ATTN_LAYERS:
        attn = model.model.language_model.layers[idx].self_attn
        shared = "shared" if attn.is_kv_shared_layer else "independent"
        store = "stores_kv" if attn.store_full_length_kv else ""
        print(f"  Layer {idx}: {shared} {store} head_dim={attn.head_dim}")

    # Prepare input — use real text from FineWeb-Edu
    print(f"\nPreparing {args.seq_len}-token input...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    # Gather enough text
    texts = []
    total_tokens = 0
    for sample in ds:
        texts.append(sample["text"])
        total_tokens += len(tokenizer.encode(sample["text"], add_special_tokens=False))
        if total_tokens > args.seq_len * 2:
            break

    full_text = " ".join(texts)
    inputs = tokenizer(full_text, return_tensors="pt", max_length=args.seq_len,
                       truncation=True).to(model.device)
    actual_len = inputs["input_ids"].shape[1]
    print(f"  Actual sequence length: {actual_len}")

    # Register hooks
    print("\nRegistering KV capture hooks...")
    captured_kv, original_forwards = register_kv_hooks(model)

    # Patch attention to capture weight stats in-place
    print("Patching attention for stats capture...")
    captured_stats, original_eager, gemma4_mod = monkey_patch_eager_attention(model)

    # Forward pass
    print(f"\nRunning forward pass ({actual_len} tokens)...")
    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False, output_attentions=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  Forward pass took {elapsed:.1f}s")

    # Restore original functions
    restore_forwards(model, original_forwards)
    gemma4_mod.eager_attention_forward = original_eager

    # Analyze each full attention layer
    all_results = {
        "metadata": {
            "model_id": args.model_id,
            "seq_len": actual_len,
            "full_attention_layers": FULL_ATTN_LAYERS,
            "independent_kv_layers": INDEPENDENT_KV_LAYERS,
        },
        "layers": {},
    }

    for layer_idx in FULL_ATTN_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")
        kv_data = captured_kv.get(layer_idx)
        if kv_data is None:
            print(f"  WARNING: No KV data captured for layer {layer_idx}")
            continue

        k = kv_data["k"]  # [B, H, T, D]
        v = kv_data["v"]  # [B, H, T, D]
        is_shared = kv_data.get("is_shared", False)
        shared_from = kv_data.get("shared_from", None)

        B, H, T, D = k.shape
        print(f"  K shape: {k.shape}, V shape: {v.shape}")
        print(f"  Shared: {is_shared}" + (f" (from layer {shared_from})" if shared_from else ""))
        print(f"  K range: [{k.min():.4f}, {k.max():.4f}], mean={k.mean():.4f}")
        print(f"  V range: [{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}")

        layer_results = {
            "shape": {"B": B, "H": H, "T": T, "D": D},
            "is_shared": is_shared,
            "shared_from": shared_from,
            "k_stats": {"min": float(k.min()), "max": float(k.max()),
                        "mean": float(k.mean()), "std": float(k.std())},
            "v_stats": {"min": float(v.min()), "max": float(v.max()),
                        "mean": float(v.mean()), "std": float(v.std())},
        }

        # 2. SVD spectra
        print("  Computing SVD spectra...")
        layer_results["k_svd"] = compute_svd_spectra(k, f"layer{layer_idx}_K")
        layer_results["v_svd"] = compute_svd_spectra(v, f"layer{layer_idx}_V")

        for tensor_name, svd_key in [("K", "k_svd"), ("V", "v_svd")]:
            concat = layer_results[svd_key]["concatenated"]
            print(f"    {tensor_name} concat: 90%@rank-{concat['rank_for_90pct']}, "
                  f"95%@rank-{concat['rank_for_95pct']}, 99%@rank-{concat['rank_for_99pct']}")

        # 3. Per-head cosine similarity (K only — V follows same pattern)
        print("  Computing head cosine similarity...")
        layer_results["k_head_similarity"] = compute_head_cosine_similarity(k)
        print(f"    K mean off-diag cosine: {layer_results['k_head_similarity']['mean_off_diagonal']:.4f}")

        # 4. Temporal deltas
        print("  Computing temporal deltas...")
        layer_results["k_temporal"] = compute_temporal_deltas(k, "K")
        layer_results["v_temporal"] = compute_temporal_deltas(v, "V")
        print(f"    K mean delta: {layer_results['k_temporal']['overall_mean_delta']:.4f}")
        print(f"    V mean delta: {layer_results['v_temporal']['overall_mean_delta']:.4f}")

        # 5. Attention weight stats (if captured)
        if layer_idx in captured_stats:
            print("  Attention weight stats (captured in-place)...")
            layer_results["attention_stats"] = captured_stats[layer_idx]
            print(f"    Mean entropy: {layer_results['attention_stats']['overall_mean_entropy']:.4f}")
        else:
            print("  No attention weights captured for this layer")

        # 6. Low-rank reconstruction
        print(f"  Testing low-rank reconstruction (ranks={args.low_rank_tests})...")
        layer_results["k_low_rank"] = compute_low_rank_reconstruction(k, args.low_rank_tests)
        layer_results["v_low_rank"] = compute_low_rank_reconstruction(v, args.low_rank_tests)
        for rank in args.low_rank_tests:
            k_err = layer_results["k_low_rank"][rank]["relative_error"]
            v_err = layer_results["v_low_rank"][rank]["relative_error"]
            k_exp = layer_results["k_low_rank"][rank]["explained_variance"]
            v_exp = layer_results["v_low_rank"][rank]["explained_variance"]
            print(f"    rank-{rank}: K err={k_err:.4f} ({k_exp:.1%}), V err={v_err:.4f} ({v_exp:.1%})")

        # Save raw KV if requested
        if args.save_raw_kv:
            raw_path = os.path.join(args.output_dir, f"layer_{layer_idx}_kv.pt")
            torch.save({"k": k.half(), "v": v.half()}, raw_path)
            print(f"  Saved raw KV to {raw_path}")

        all_results["layers"][str(layer_idx)] = layer_results

        # Free memory
        del k, v
        gc.collect()

    # Save JSON results
    json_path = os.path.join(args.output_dir, f"kv_analysis_seq{actual_len}.json")

    # Convert any remaining numpy/torch types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n=== Results saved to {json_path} ===")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'Layer':>6} {'Shared':>7} {'K 95%':>8} {'V 95%':>8} "
          f"{'K cos':>7} {'K delta':>8} {'V delta':>8} "
          f"{'K r64 err':>10} {'V r64 err':>10}")
    print("-" * 95)
    for layer_idx in FULL_ATTN_LAYERS:
        key = str(layer_idx)
        if key not in all_results["layers"]:
            continue
        r = all_results["layers"][key]
        shared = "yes" if r["is_shared"] else "no"
        k95 = r["k_svd"]["concatenated"]["rank_for_95pct"]
        v95 = r["v_svd"]["concatenated"]["rank_for_95pct"]
        kcos = r["k_head_similarity"]["mean_off_diagonal"]
        kdelta = r["k_temporal"]["overall_mean_delta"]
        vdelta = r["v_temporal"]["overall_mean_delta"]
        k64 = r["k_low_rank"].get(64, r["k_low_rank"].get(str(64), {}))
        v64 = r["v_low_rank"].get(64, r["v_low_rank"].get(str(64), {}))
        k64_err = k64.get("relative_error", 0) if k64 else 0
        v64_err = v64.get("relative_error", 0) if v64 else 0
        print(f"{layer_idx:>6} {shared:>7} {k95:>8} {v95:>8} "
              f"{kcos:>7.4f} {kdelta:>8.4f} {vdelta:>8.4f} "
              f"{k64_err:>10.4f} {v64_err:>10.4f}")


if __name__ == "__main__":
    main()
