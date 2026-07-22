import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationConfig:
    model: str
    dataset: str
    backend: str
    prompt_type: int
    max_tokens: int
    sample_size: int | None
    sample_range: tuple[int, int] | None
    sample_indices: list[int] | None
    from_pickle: str | None
    from_pregenerated: str | None
    discord: bool
    tag: str | None
    debug_nocache: bool
    experimental_llama_batch: bool
    api_concurrency: int = 1
    api_datapoint_retries: int = 3
    api_retry_initial_delay: float = 5.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GenerationConfig":
        return cls(
            model=args.model,
            dataset=args.dataset,
            backend=args.backend,
            prompt_type=args.prompt_type,
            max_tokens=args.max_tokens,
            sample_size=args.sample_size,
            sample_range=tuple(args.sample_range) if args.sample_range else None,
            sample_indices=cls._load_sample_indices(args.sample_indices),
            from_pickle=args.from_pickle,
            from_pregenerated=args.from_pregenerated,
            discord=args.discord if args.discord else False,
            tag=args.tag,
            debug_nocache=args.debug_nocache,
            experimental_llama_batch=args.experimental_llama_batch,
            api_concurrency=args.api_concurrency,
            api_datapoint_retries=args.api_datapoint_retries,
            api_retry_initial_delay=args.api_retry_initial_delay,
        )

    @staticmethod
    def _load_sample_indices(path: str | None) -> list[int] | None:
        if path is None:
            return None
        with open(path) as f:
            return [int(line.strip()) for line in f if line.strip()]



@dataclass
class ConfidenceConfig:
    confidence: str
    nb_stepbootstrap_samples: int | None
    debug_top20: bool
    cuda_sync_for_timing: bool
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ConfidenceConfig":
        return cls(
            confidence=args.confidence,
            nb_stepbootstrap_samples=args.nb_stepbootstrap_samples,
            debug_top20=args.debug_top20,
            cuda_sync_for_timing=args.experimental_cuda_sync_for_timing,
        )



@dataclass
class SamplingConfig:
    temperature: float
    nb_cot_samples: int
    nb_stepbootstrap_samples: int
    seed_stepbootstrap: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SamplingConfig":
        return cls(
            temperature=args.temperature,
            nb_cot_samples=args.nb_cot_samples,
            nb_stepbootstrap_samples=args.nb_stepbootstrap_samples,
            seed_stepbootstrap=args.seed_stepbootstrap,
        )




