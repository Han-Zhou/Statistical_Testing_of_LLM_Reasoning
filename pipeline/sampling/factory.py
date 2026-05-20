

from abc import ABC, abstractmethod

from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration
from config import GenerationConfig, SamplingConfig

from pipeline.sampling.context import SampleContext
from pipeline.sampling.vanilla_sampling import VanillaSampling
from pipeline.sampling.rejection_sampling import RejectionSampling
from pipeline.sampling.lawyer_sampling import LawyerSampling
from pipeline.sampling.stepbootstrap_sampling import StepBootstrapSampling

from .base import SamplingMethod



class SamplingMethodFactory:
    @staticmethod
    def create(generation_config: GenerationConfig, sampling_config: SamplingConfig, context: SampleContext) -> list[SamplingMethod]:
        return [
            VanillaSampling(generation_config, sampling_config, context),
            RejectionSampling(generation_config, sampling_config, context),
            LawyerSampling(generation_config, sampling_config, context),
            StepBootstrapSampling(generation_config, sampling_config, context)
        ]

