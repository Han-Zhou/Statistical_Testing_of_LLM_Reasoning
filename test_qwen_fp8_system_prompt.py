"""Run Qwen FP8 vanilla generation with an editable system prompt.

This is a GPU script: it loads the registered Qwen FP8 checkpoint and writes
only ``trajectories/<run_name>/vanilla``.  Rejection, lawyer, and
step-bootstrap trajectories are not generated.
"""

from __future__ import annotations

import argparse
import logging
import os

from config import ConfidenceConfig, GenerationConfig, SamplingConfig
from datasets.bigbench_movie import BigBenchMovieDataset
from domain import Datapoint, PromptRequest
from pipeline import Runner
from repository import TrajectoryRepository


# Edit this string to test a different system prompt.  Keeping it here avoids
# changing the prompt used by normal BigBench Movie runs.
# SYSTEM_PROMPT = """\
# You are an expert reasoning assistant. For every problem you receive, think \
# carefully and reason step-by-step. Label each reasoning step as 'Step 1:', \
# 'Step 2:', etc. Keep your reasoning concise, and be brief in each step.

# When you output the final answer, output ONLY the single letter corresponding \
# to the correct answer. Do not include periods, explanations, or any other text.

# During your reasoning, do NOT reveal, hint at, or restate the final answer. Do \
# not write lines like 'Answer:', 'Final answer:', any answer strings, or any \
# concluding sentence. Stop immediately after your last numbered reasoning step.
# """

SYSTEM_PROMPT = """\
You are an expert reasoning assistant. For every problem you receive, think \
carefully and reason step-by-step. Label each reasoning step as 'Step 1:', \
'Step 2:', etc. Keep your reasoning concise, and be brief in each step.

When you output the final answer, output ONLY the single letter corresponding \
to the correct answer. Do not include periods, explanations, or any other text.
"""


DEFAULT_PICKLE_PATH = os.environ.get(
    "QWEN_FP8_PICKLE_PATH",
    "/shared_work/han/storage/cot/pickles/bigbench_movie_250.pkl",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class SystemPromptBigBenchMovieDataset(BigBenchMovieDataset):
    """BigBench Movie dataset using this script's ``SYSTEM_PROMPT``."""

    def build_messages(
        self,
        datapoint: Datapoint,
        prompt_request: PromptRequest,
    ) -> list[dict[str, str]]:
        messages = super().build_messages(datapoint, prompt_request)
        if not messages or messages[0].get("role") != "system":
            raise RuntimeError(
                "Expected BigBench Movie messages to start with a system prompt"
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            *messages[1:],
        ]


class VanillaOnlyQwenFp8Runner(Runner):
    """Qwen FP8 runner that creates and executes only the vanilla repository."""

    def __init__(
        self,
        generation_config: GenerationConfig,
        confidence_config: ConfidenceConfig,
        sampling_config: SamplingConfig,
        discord: bool = False,
    ) -> None:
        super().__init__(
            generation_config=generation_config,
            confidence_config=confidence_config,
            sampling_config=sampling_config,
            discord=discord,
        )
        self.dataset = SystemPromptBigBenchMovieDataset()
        self.context.dataset = self.dataset

    def _init_repositories(self, samples: str | int) -> None:
        sample_suffix = str(samples)
        if not sample_suffix.startswith("s"):
            sample_suffix = f"s{sample_suffix}"
        base_dir_name = (
            f"trajectories/{self.generation_config.tag}_qwen_fp8_"
            f"bigbench_movie_{sample_suffix}"
        )
        self.vanilla_trajectory_repository = TrajectoryRepository(
            f"{base_dir_name}/vanilla"
        )

    def run_generation_and_confidence(self, datapoint: Datapoint) -> None:
        self.context.datapoint = datapoint
        try:
            self._run_generation_and_confidence_vanilla()
        finally:
            self.context.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only Qwen FP8 vanilla generation on BigBench Movie using the "
            "editable SYSTEM_PROMPT near the top of this file."
        )
    )
    parser.add_argument(
        "--from_pickle",
        default=DEFAULT_PICKLE_PATH,
        help=(
            "BigBench Movie pickle path. Defaults to QWEN_FP8_PICKLE_PATH, "
            "then the machine-local test pickle."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample_size", type=int, default=1)
    selection.add_argument(
        "--sample_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
    )
    selection.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Path containing one dataset index per line.",
    )
    parser.add_argument("--max_tokens", type=int, default=5024)
    parser.add_argument("--prompt_type", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tag", default="test-qwen-fp8-system-prompt")
    parser.add_argument("--debug_top20", action="store_true")
    parser.add_argument("--experimental_cuda_sync_for_timing", action="store_true")
    parser.add_argument("--discord", action="store_true")
    args = parser.parse_args()

    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample_size must be at least 1")
    if (
        args.sample_range is not None
        and args.sample_range[0] >= args.sample_range[1]
    ):
        parser.error("--sample_range START must be smaller than END")
    if args.max_tokens < 1:
        parser.error("--max_tokens must be at least 1")
    if not args.tag:
        parser.error("--tag must not be empty")
    return args


def main() -> None:
    args = parse_args()
    generation_config = GenerationConfig(
        model="qwen_fp8",
        dataset="bigbench_movie",
        backend="hf",
        prompt_type=args.prompt_type,
        max_tokens=args.max_tokens,
        sample_size=args.sample_size,
        sample_range=tuple(args.sample_range) if args.sample_range else None,
        sample_indices=GenerationConfig._load_sample_indices(args.sample_indices),
        from_pickle=args.from_pickle,
        from_pregenerated=None,
        discord=args.discord,
        tag=args.tag,
        debug_nocache=False,
        experimental_llama_batch=False,
    )
    confidence_config = ConfidenceConfig(
        confidence="vanilla",
        nb_stepbootstrap_samples=None,
        debug_top20=args.debug_top20,
        cuda_sync_for_timing=args.experimental_cuda_sync_for_timing,
    )
    sampling_config = SamplingConfig(
        temperature=0.0,
        nb_cot_samples=1,
        nb_stepbootstrap_samples=0,
        seed_stepbootstrap=0,
    )

    runner = VanillaOnlyQwenFp8Runner(
        generation_config=generation_config,
        confidence_config=confidence_config,
        sampling_config=sampling_config,
        discord=args.discord,
    )
    runner.run()


if __name__ == "__main__":
    main()
