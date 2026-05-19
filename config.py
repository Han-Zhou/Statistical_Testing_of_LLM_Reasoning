import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationConfig:
    model: str
    dataset: str
    backend: str
    thinking: bool
    prompt_type: int
    max_new_tokens: int
    sample_size: int | None
    sample_range: tuple[int, int] | None
    from_pregenerated: str | None
    discord: bool
    tag: str | None

    @property
    def from_args(args: argparse.Namespace) -> "GenerationConfig":
        return GenerationConfig(
            model=args.model,
            dataset=args.dataset,
            backend=args.backend,
            thinking=args.thinking,
            prompt_type=args.prompt_type,
            max_new_tokens=args.max_new_tokens,
            sample_size=args.sample_size,
            sample_range=tuple(args.sample_range) if args.sample_range else None,
            from_pregenerated=args.from_pregenerated,
            discord=args.discord if args.discord else False,
            tag=args.tag,
        )



@dataclass
class ConfidenceConfig:
    confidence: str
    nb_dropout_samples: int | None

    @property
    def from_args(args: argparse.Namespace) -> "ConfidenceConfig":
        return ConfidenceConfig(
            confidence=args.confidence,
            nb_dropout_samples=args.nb_dropout_samples,
        )



@dataclass
class SamplingConfig:
    temperature: float
    nb_cot_samples: int

    @property
    def from_args(args: argparse.Namespace) -> "SamplingConfig":
        return SamplingConfig(
            temperature=args.temperature,
            nb_cot_samples=args.nb_cot_samples,
        )







