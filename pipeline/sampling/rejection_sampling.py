from pipeline.sampling.base import SamplingMethod

from config import GenerationConfig, SamplingConfig
from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration, PromptRequest


class RejectionSampling(SamplingMethod):
    def generate(self, datapoint: Datapoint) -> ParsedOutputGeneration:
        ...