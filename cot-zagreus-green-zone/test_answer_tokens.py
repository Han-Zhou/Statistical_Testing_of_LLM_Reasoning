import argparse

from transformers import AutoTokenizer

from models.adapters.registry import ANSWER_TOKENS
from models.core_models.registry import MODEL_HF_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Show tokenized representations of ANSWER_TOKENS for a given model.")
    parser.add_argument("--model", default="llama", help=f"Model key in MODEL_HF_REGISTRY. Options: {list(MODEL_HF_REGISTRY.keys())}")
    args = parser.parse_args()

    if args.model not in MODEL_HF_REGISTRY:
        raise ValueError(f"Model '{args.model}' not found in registry. Options: {list(MODEL_HF_REGISTRY.keys())}")

    actual_model_name = MODEL_HF_REGISTRY[args.model]
    print(f"Loading tokenizer for: {args.model} ({actual_model_name})\n")
    tokenizer = AutoTokenizer.from_pretrained(actual_model_name)

    for category, tokens in ANSWER_TOKENS.items():
        print(f"=== {category!r} ({len(tokens)} entries) ===")
        header = f"{'string':<20} {'ids':<30} {'pieces'}"
        print(header)
        print("-" * len(header))
        for s in tokens:
            ids = tokenizer(s, add_special_tokens=False).input_ids
            pieces = tokenizer.convert_ids_to_tokens(ids)
            single_marker = "" if len(ids) == 1 else "  <-- multi-token"
            print(f"{repr(s):<20} {str(ids):<30} {pieces}{single_marker}")
        print()


if __name__ == "__main__":
    main()
