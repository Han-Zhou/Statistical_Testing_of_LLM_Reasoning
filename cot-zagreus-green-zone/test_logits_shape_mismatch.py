"""
Demonstrates the shape mismatch between `model.generate(..., output_logits=True,
return_dict_in_generate=True)` and `model(input_ids=...)` (a.k.a. forward).

Both APIs return an object with `.logits`, but the shapes are completely
different — and `process_generation_output` in models/adapters/llama_adapter.py
assumes the generate-path shape. When stepbootstrap funnels forward-path output
into the same parser, `all_probs` ends up with shape [1, T_delta, vocab]
instead of [T_generated, vocab], `num_question_tokens` is computed wrongly,
and the answer-probs slice comes out empty.

Run:   python test_logits_shape_mismatch.py
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.core_models.registry import MODEL_HF_REGISTRY


def show(label, t):
    if isinstance(t, tuple):
        print(f"  {label:<35} type=tuple  len={len(t)}  elem.shape={tuple(t[0].shape)}")
    else:
        print(f"  {label:<35} type=Tensor shape={tuple(t.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    actual = MODEL_HF_REGISTRY[args.model]
    print(f"Loading {args.model} ({actual})\n")
    tokenizer = AutoTokenizer.from_pretrained(actual)
    model = AutoModelForCausalLM.from_pretrained(
        actual, device_map="auto", dtype=torch.bfloat16
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # ---------- generate path ----------
    print("=" * 72)
    print("model.generate(..., output_logits=True, return_dict_in_generate=True)")
    print("=" * 72)
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"  prompt_len           = {prompt_len}")
    print(f"  sequences.shape      = {tuple(gen_out.sequences.shape)}")
    show("logits", gen_out.logits)

    # The exact comprehension from llama_adapter.py:229
    gen_all_probs = torch.stack(
        [F.softmax(s, dim=-1).squeeze(0) for s in gen_out.logits]
    )
    print()
    print("  After `torch.stack([F.softmax(s, dim=-1).squeeze(0) for s in outputs.logits])`:")
    show("all_probs", gen_all_probs)

    output_tokens_gen = tokenizer.convert_ids_to_tokens(gen_out.sequences[0])
    num_q_gen = len(output_tokens_gen) - gen_all_probs.shape[0]
    print(f"  len(output_tokens)   = {len(output_tokens_gen)}")
    print(f"  num_question_tokens  = {num_q_gen}   (expected ≈ prompt_len = {prompt_len})")

    # ---------- forward path ----------
    print()
    print("=" * 72)
    print("model(input_ids=..., return_dict=True)   # the 'forward' path")
    print("=" * 72)
    with torch.inference_mode():
        fwd_out = model(input_ids=inputs.input_ids, return_dict=True, use_cache=False)
    print(f"  input_ids.shape      = {tuple(inputs.input_ids.shape)}")
    show("logits", fwd_out.logits)

    fwd_all_probs = torch.stack(
        [F.softmax(s, dim=-1).squeeze(0) for s in fwd_out.logits]
    )
    print()
    print("  After the SAME comprehension:")
    show("all_probs", fwd_all_probs)
    print(f"  ^ note the leading singleton dim — iterating a [1, T, vocab] tensor")
    print(f"    yields ONE element of shape [T, vocab], so squeeze(0) is a no-op")
    print(f"    and stack() re-adds a length-1 axis.")

    # Pretend we (wrongly) treat fwd_all_probs as [T, vocab] like the parser does.
    # On the forward path, outputs.sequences doesn't exist by default — this is
    # what the patched LLM.forward_pass attaches manually. We simulate it here.
    output_tokens_fwd = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    num_q_fwd = len(output_tokens_fwd) - fwd_all_probs.shape[0]
    print()
    print(f"  len(output_tokens)   = {len(output_tokens_fwd)}")
    print(f"  all_probs.shape[0]   = {fwd_all_probs.shape[0]}   (the WRONG axis)")
    print(f"  num_question_tokens  = {num_q_fwd}   (expected ≈ {prompt_len - args.max_new_tokens} or 0;")
    print(f"                                       instead it's len(seq) - 1, way off)")

    # Show what _extract_answer_and_probs would slice. Pick the last token as a
    # stand-in for an answer position.
    fake_answer_start = len(output_tokens_fwd) - 2
    fake_answer_end = len(output_tokens_fwd)
    sliced = fwd_all_probs[fake_answer_start - num_q_fwd : fake_answer_end - num_q_fwd]
    print()
    print(f"  Simulated answer-probs slice: all_probs[{fake_answer_start - num_q_fwd}:{fake_answer_end - num_q_fwd}]")
    print(f"  → result shape = {tuple(sliced.shape)}, numel = {sliced.numel()}")
    print(f"  ^ THIS is why answer_probabilities ends up [] in the trajectory JSON.")

    # ---------- what the forward path SHOULD produce ----------
    print()
    print("=" * 72)
    print("What the forward path SHOULD do to mirror the generate-path semantics")
    print("=" * 72)
    print("  generate-path all_probs[k] = distribution that produced generated token k")
    print("  forward-path  logits[0, i] = distribution predicting token i+1")
    print()
    print("  To get an [T_generated, vocab] tensor where row k = distribution that")
    print("  produced token (prompt_len + k), do:")
    print()
    print("      all_probs = F.softmax(outputs.logits[0, prompt_len-1:-1, :], dim=-1)")
    print()
    fwd_aligned = F.softmax(fwd_out.logits[0, prompt_len - 1 : -1, :], dim=-1)
    show("aligned all_probs", fwd_aligned)
    print(f"  ^ shape[0] now matches the number of post-prompt tokens, not 1.")


if __name__ == "__main__":
    main()
