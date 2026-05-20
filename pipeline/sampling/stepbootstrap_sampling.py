from pipeline.sampling.base import SamplingMethod

from config import GenerationConfig, SamplingConfig
from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration, PromptRequest


class StepBootstrapSampling(SamplingMethod):
    def generate(self, datapoint: Datapoint) -> ParsedOutputGeneration:
        ...