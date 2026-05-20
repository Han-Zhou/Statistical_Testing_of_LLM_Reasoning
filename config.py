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
    max_tokens: int
    sample_size: int | None
    sample_range: tuple[int, int] | None
    from_pickle: str | None
    from_pregenerated: str | None
    discord: bool
    tag: str | None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GenerationConfig":
        return cls(
            model=args.model,
            dataset=args.dataset,
            backend=args.backend,
            thinking=args.thinking,
            prompt_type=args.prompt_type,
            max_tokens=args.max_tokens,
            sample_size=args.sample_size,
            sample_range=tuple(args.sample_range) if args.sample_range else None,
            from_pickle=args.from_pickle,
            from_pregenerated=args.from_pregenerated,
            discord=args.discord if args.discord else False,
            tag=args.tag,
        )



@dataclass
class ConfidenceConfig:
    confidence: str
    nb_dropout_samples: int | None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ConfidenceConfig":
        return cls(
            confidence=args.confidence,
            nb_dropout_samples=args.nb_dropout_samples,
        )



@dataclass
class SamplingConfig:
    temperature: float
    nb_cot_samples: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SamplingConfig":
        return cls(
            temperature=args.temperature,
            nb_cot_samples=args.nb_cot_samples,
        )







