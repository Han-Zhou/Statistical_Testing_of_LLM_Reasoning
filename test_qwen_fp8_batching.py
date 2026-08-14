"""GPU diagnostic for Qwen3.6 FP8 homogeneous vs heterogeneous batching.

This is intentionally a standalone script rather than a unit test: loading the
27B checkpoint requires CUDA and can take several minutes.  By default it runs
the potentially failing heterogeneous case last, after serial and same-length
batch controls have completed.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from models.adapters.qwen_adapter import QwenFp8Adapter


FILLER = (
    "Step: compare the evidence carefully, preserve every relevant detail, "
    "and check the conclusion before continuing.\n"
)


class NonFiniteLogitsError(RuntimeError):
    """Raised before sampling when a model row contains NaN or infinity."""


class FiniteLogitsProcessor(LogitsProcessor):
    """Validate raw next-token logits on every decoding step.

    The check runs before temperature/top-k/top-p processing.  This adds a CUDA
    synchronization per token and is therefore for diagnosis only, not normal
    inference.
    """

    def __init__(self, case_name: str) -> None:
        self.case_name = case_name
        self.step = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self.step += 1
        finite = torch.isfinite(scores)
        if bool(finite.all().item()):
            return scores

        bad_per_row = (~finite).sum(dim=-1)
        bad_rows = bad_per_row.nonzero(as_tuple=False).flatten().tolist()
        row_details = []
        for row in bad_rows:
            row_scores = scores[row]
            row_finite = finite[row]
            finite_scores = row_scores[row_finite]
            minimum = float(finite_scores.min().item()) if finite_scores.numel() else None
            maximum = float(finite_scores.max().item()) if finite_scores.numel() else None
            row_details.append(
                f"row={row}, invalid={int(bad_per_row[row].item())}, "
                f"finite_min={minimum}, finite_max={maximum}"
            )

        raise NonFiniteLogitsError(
            f"{self.case_name}: non-finite raw logits at generated token "
            f"{self.step}; " + "; ".join(row_details)
        )


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    elapsed_seconds: float
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare serial, same-length batched, and different-length "
            "left-padded generation with Qwen3.6 FP8."
        )
    )
    parser.add_argument(
        "--case",
        choices=("all", "serial", "homogeneous", "heterogeneous"),
        default="all",
        help="Diagnostic case to run. 'all' runs the risky heterogeneous case last.",
    )
    parser.add_argument(
        "--short-repeats",
        type=int,
        default=32,
        help="Number of filler reasoning steps in the shorter prompt.",
    )
    parser.add_argument(
        "--long-repeats",
        type=int,
        default=128,
        help="Number of filler reasoning steps in the longer prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum generated tokens per case; 200 matches Qwen phase 3.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.short_repeats < 0 or args.long_repeats < 0:
        parser.error("prompt repeat counts must be non-negative")
    if args.short_repeats >= args.long_repeats:
        parser.error("--short-repeats must be smaller than --long-repeats")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0 to exercise sampling")
    return args


def build_phase_3_prompt(adapter: QwenFp8Adapter, repeats: int) -> str:
    """Build a text-only prompt shaped like the adapter's phase-3 input."""
    prompt = adapter.render_prompt(
        [
            {
                "role": "user",
                "content": (
                    "Which movie is described by the reasoning below? Return a "
                    "short final answer."
                ),
            }
        ]
    )
    return prompt + (FILLER * repeats) + "\nThe answer is \\boxed{"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_generation_case(
    adapter: QwenFp8Adapter,
    name: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> CaseResult:
    wrapper = adapter.model
    tokenizer = wrapper.tokenizer
    set_seed(seed)

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left" if len(prompts) > 1 else "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=len(prompts) > 1,
        add_special_tokens=False,
    ).to(wrapper.input_device)
    prompt_lengths = inputs.attention_mask.sum(dim=-1).tolist()
    left_padding = [inputs.input_ids.shape[1] - length for length in prompt_lengths]
    print(
        f"\n[{name}] rows={len(prompts)}, prompt_tokens={prompt_lengths}, "
        f"left_padding={left_padding}",
        flush=True,
    )

    checker = FiniteLogitsProcessor(name)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            outputs = wrapper.model.generate(
                **inputs,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([checker]),
            )
        # Synchronize here so an asynchronous model-kernel failure is attributed
        # to this case rather than to the next diagnostic.
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated_counts = [
            int(outputs.sequences.shape[1] - inputs.input_ids.shape[1])
            for _ in prompts
        ]
        detail = (
            f"checked {checker.step} sampling steps; "
            f"generated_tokens={generated_counts}"
        )
        print(f"[{name}] PASS: {detail} ({elapsed:.1f}s)", flush=True)
        return CaseResult(name, True, elapsed, detail)
    except Exception as error:
        elapsed = time.perf_counter() - start
        detail = f"{type(error).__name__}: {error}"
        print(f"[{name}] FAIL: {detail} ({elapsed:.1f}s)", flush=True)
        return CaseResult(name, False, elapsed, detail)
    finally:
        tokenizer.padding_side = old_padding_side


def selected_cases(case: str) -> list[str]:
    if case == "all":
        return ["serial", "homogeneous", "heterogeneous"]
    return [case]


def main() -> int:
    args = parse_args()
    print("Loading Qwen3.6-27B-FP8...", flush=True)
    adapter = QwenFp8Adapter()
    short_prompt = build_phase_3_prompt(adapter, args.short_repeats)
    long_prompt = build_phase_3_prompt(adapter, args.long_repeats)

    results: list[CaseResult] = []
    for case in selected_cases(args.case):
        if case == "serial":
            # Run both prompt lengths without padding as the control.
            for label, prompt in (
                ("serial-short", short_prompt),
                ("serial-long", long_prompt),
            ):
                result = run_generation_case(
                    adapter,
                    label,
                    [prompt],
                    args.max_new_tokens,
                    args.temperature,
                    args.seed,
                )
                results.append(result)
                if not result.passed:
                    break
        elif case == "homogeneous":
            results.append(
                run_generation_case(
                    adapter,
                    "homogeneous",
                    [long_prompt, long_prompt],
                    args.max_new_tokens,
                    args.temperature,
                    args.seed,
                )
            )
        else:
            results.append(
                run_generation_case(
                    adapter,
                    "heterogeneous",
                    [short_prompt, long_prompt],
                    args.max_new_tokens,
                    args.temperature,
                    args.seed,
                )
            )

        if not results[-1].passed:
            # A raw non-finite error is safe, but another CUDA exception may
            # have poisoned the process. Do not launch more model work.
            break

    print("\nSummary", flush=True)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status:4} {result.name}: {result.detail}", flush=True)

    if all(result.passed for result in results):
        print(
            "No failure was reproduced. Increase --max-new-tokens or the "
            "prompt repeat counts before concluding the path is stable.",
            flush=True,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
