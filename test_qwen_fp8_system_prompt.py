"""Run distributed Qwen FP8 vanilla generation with an editable system prompt.

This is a GPU script: it loads the registered Qwen FP8 checkpoint and writes
only ``trajectories/<run_name>/vanilla``.  Rejection, lawyer, and
step-bootstrap trajectories are not generated.  Launch one process per GPU:

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
        test_qwen_fp8_system_prompt.py --sample_size 1
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from config import ConfidenceConfig, GenerationConfig, SamplingConfig
from datasets.bigbench_movie import BigBenchMovieDataset
from domain import Datapoint, PromptRequest
from models.adapters.qwen_adapter import QwenFp8Adapter, QwenScorer
from models.core_models import LLM
from pipeline import Runner
import pipeline.runner as runner_module
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
    "/home/han/workspace/data_cot/pickles/bigbench_movie_250.pkl",
)

MODEL_ID = "Qwen/Qwen3.8-27B-FP8"

# This is the same native Transformers plan validated by
# temp_test_qwen_fp8_phase1.py.  The automatic plan leaves the embedding and
# Qwen3.5 linear-attention projections replicated, which exceeds 16 GiB per
# rank.  Module-level rules also shard each FP8 ``weight_scale_inv`` alongside
# its corresponding weight.
CUSTOM_TP_PLAN: dict[str, str] = {
    "model.embed_tokens": "embedding_rowwise",
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
    "model.layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_z": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_b": "colwise_gather_output",
    "model.layers.*.linear_attn.in_proj_a": "colwise_gather_output",
    "model.layers.*.linear_attn.out_proj": "rowwise_split_input",
    "lm_head": "colwise_gather_output",
}


def _distributed_value(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from error


RANK = _distributed_value("RANK", 0)
LOCAL_RANK = _distributed_value("LOCAL_RANK", 0)
WORLD_SIZE = _distributed_value("WORLD_SIZE", 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _patch_fp8_config(config: Any) -> dict[str, int]:
    """Adapt the multimodal checkpoint's FP8 exclusions to its text model."""

    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        raise RuntimeError("Checkpoint has no quantization_config metadata.")

    if isinstance(quantization_config, dict):
        excluded = quantization_config.get("modules_to_not_convert", [])
    else:
        excluded = getattr(quantization_config, "modules_to_not_convert", [])
    if excluded is None:
        excluded = []
    if not isinstance(excluded, (list, tuple)) or not all(
        isinstance(name, str) for name in excluded
    ):
        raise RuntimeError("FP8 modules_to_not_convert must be a list of strings.")

    corrected: list[str] = []
    removed_gate = 0
    removed_non_text = 0
    rewritten_text_prefix = 0
    for name in excluded:
        if name.endswith(".mlp.gate"):
            removed_gate += 1
            continue
        if (
            name in {"visual", "model.visual"}
            or name.startswith(("visual.", "model.visual.", "mtp."))
        ):
            removed_non_text += 1
            continue
        if name.startswith("model.language_model."):
            name = "model." + name.removeprefix("model.language_model.")
            rewritten_text_prefix += 1
        corrected.append(name)

    if isinstance(quantization_config, dict):
        quantization_config["modules_to_not_convert"] = corrected
    else:
        quantization_config.modules_to_not_convert = corrected
    return {
        "original": len(excluded),
        "remaining": len(corrected),
        "removed_mlp_gate": removed_gate,
        "removed_non_text": removed_non_text,
        "rewritten_text_prefix": rewritten_text_prefix,
    }


def _validate_loading_info(loading_info: Any) -> None:
    """Refuse inference if any text FP8 scale failed to load."""

    if not isinstance(loading_info, dict):
        raise RuntimeError("from_pretrained returned malformed loading information.")

    rejected_scales: list[str] = []
    for category in ("unexpected_keys", "missing_keys", "mismatched_keys"):
        entries = loading_info.get(category, []) or []
        keys = [
            str(entry[0] if isinstance(entry, (list, tuple)) and entry else entry)
            for entry in entries
        ]
        rejected_scales.extend(
            f"{category}: {key}"
            for key in keys
            if "weight_scale_inv" in key
            and not key.startswith(("model.visual.", "visual.", "mtp."))
        )
        if RANK == 0:
            logging.info("%s_count=%d sample=%s", category, len(keys), keys[:5])

    if rejected_scales:
        raise RuntimeError(
            "FP8 scale tensors were not loaded correctly; refusing to generate.\n"
            + "\n".join(rejected_scales[:10])
        )


class DistributedQwenFp8LLM(LLM):
    """Repository LLM wrapper using the validated native-HF TP loader."""

    def _load_qwen_fp8_model(self, actual_model_name: str) -> None:
        if actual_model_name != MODEL_ID:
            raise RuntimeError(
                f"Expected the registered checkpoint {MODEL_ID}, got {actual_model_name}."
            )
        if WORLD_SIZE != 2:
            raise RuntimeError(
                "Launch with exactly two processes: CUDA_VISIBLE_DEVICES=0,1 "
                "torchrun --standalone --nproc-per-node=2 "
                "test_qwen_fp8_system_prompt.py --sample_size 1"
            )
        if not torch.cuda.is_available() or LOCAL_RANK >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid CUDA setup: local_rank={LOCAL_RANK}, "
                f"device_count={torch.cuda.device_count()}."
            )

        torch.cuda.set_device(LOCAL_RANK)
        common_kwargs = {"trust_remote_code": True}
        config = AutoConfig.from_pretrained(actual_model_name, **common_kwargs)
        patch_counts = _patch_fp8_config(config)
        if RANK == 0:
            free_bytes, total_bytes = torch.cuda.mem_get_info(LOCAL_RANK)
            logging.info(
                "Loading custom TP FP8 model: free=%.3f GiB total=%.3f GiB "
                "config_patch=%s plan_styles=%s",
                free_bytes / 2**30,
                total_bytes / 2**30,
                patch_counts,
                dict(Counter(CUSTOM_TP_PLAN.values())),
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_name,
            **common_kwargs,
        )

        # Previous single-process/CPU-offload loader retained for reference:
        # from transformers import Qwen3_5ForConditionalGeneration
        # self.model, loading_info = (
        #     Qwen3_5ForConditionalGeneration.from_pretrained(
        #         actual_model_name,
        #         config=config,
        #         device_map="balanced_low_0",
        #         max_memory={0: "11GiB", 1: "14GiB", "cpu": "100GiB"},
        #         offload_buffers=True,
        #         dtype="auto",
        #         output_loading_info=True,
        #         trust_remote_code=True,
        #         allow_all_kernels=True,
        #     )
        # )

        self.model, loading_info = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            config=config,
            tp_plan=CUSTOM_TP_PLAN,
            dtype="auto",
            output_loading_info=True,
            allow_all_kernels=True,
            key_mapping={
                r"^model\.language_model\.(.+)$": r"model.\1",
            },
            **common_kwargs,
        )
        _validate_loading_info(loading_info)
        self.model.eval()

        if getattr(self.model, "_tp_size", None) != 2:
            raise RuntimeError("Transformers did not initialize two-way TP.")
        if getattr(self.model, "tp_plan", None) != CUSTOM_TP_PLAN:
            raise RuntimeError("Transformers did not retain the custom TP plan.")

        local_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        if RANK == 0:
            logging.info(
                "Custom TP model loaded: class=%s local_parameters=%.3f GiB",
                type(self.model).__name__,
                local_parameter_bytes / 2**30,
            )


class DistributedQwenFp8Adapter(QwenFp8Adapter):
    """Qwen adapter that avoids the registry's single-process FP8 loader."""

    def __init__(self) -> None:
        self.enable_thinking = True
        self.model = DistributedQwenFp8LLM(
            model_name="qwen_fp8",
            attention_implementation=None,
        )
        self.model_scorer = QwenScorer(self.model)


class _RankLocalNullRepository:
    """Prevent nonzero TP ranks from racing rank 0's trajectory writes."""

    def save(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_error(self, *args: Any, **kwargs: Any) -> None:
        return None


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
        # Previous single-process adapter construction retained for reference:
        # super().__init__(
        #     generation_config=generation_config,
        #     confidence_config=confidence_config,
        #     sampling_config=sampling_config,
        #     discord=discord,
        # )

        # Runner resolves the adapter through a module-level registry. Replace
        # only its test-local qwen_fp8 entry while super().__init__ constructs
        # the normal confidence and sampling pipeline, then restore it.
        original_registry = runner_module.MODEL_ADAPTER_REGISTRY
        runner_module.MODEL_ADAPTER_REGISTRY = {
            **original_registry,
            "qwen_fp8": DistributedQwenFp8Adapter,
        }
        try:
            super().__init__(
                generation_config=generation_config,
                confidence_config=confidence_config,
                sampling_config=sampling_config,
                discord=discord and RANK == 0,
            )
        finally:
            runner_module.MODEL_ADAPTER_REGISTRY = original_registry

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

        # Previous single-process repository construction retained for reference:
        # self.vanilla_trajectory_repository = TrajectoryRepository(
        #     f"{base_dir_name}/vanilla"
        # )
        if RANK == 0:
            self.vanilla_trajectory_repository = TrajectoryRepository(
                f"{base_dir_name}/vanilla"
            )
        else:
            self.vanilla_trajectory_repository = _RankLocalNullRepository()

    def run_generation_and_confidence(self, datapoint: Datapoint) -> None:
        self.context.datapoint = datapoint
        try:
            self._run_generation_and_confidence_vanilla()
        finally:
            self.context.clear()
        if dist.is_initialized():
            dist.barrier(device_ids=[LOCAL_RANK])

    def _run_datapoint_with_retries(self, datapoint: Datapoint) -> None:
        """Propagate rank failures so torchrun can terminate the peer rank."""

        self.run_generation_and_confidence(datapoint)


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

    args.max_tokens = 1024

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

    try:
        runner = VanillaOnlyQwenFp8Runner(
            generation_config=generation_config,
            confidence_config=confidence_config,
            sampling_config=sampling_config,
            discord=args.discord,
        )
        runner.run()
        if dist.is_initialized():
            dist.barrier(device_ids=[LOCAL_RANK])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
