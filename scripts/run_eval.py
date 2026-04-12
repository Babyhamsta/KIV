"""Quick smoke test for KIV middleware."""
import sys
import os
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=== KIV Smoke Test ===")
    print("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    model.eval()

    from kiv import KIVConfig, KIVMiddleware
    from kiv.eval_harness import run_all_tests

    config = KIVConfig(hot_budget=2048, top_p=256)
    run_all_tests(model, tokenizer, config)

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()
