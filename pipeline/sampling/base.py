

from abc import ABC, abstractmethod

from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration
from config import GenerationConfig, SamplingConfig

from pipeline.sampling.context import SampleContext





class SamplingMethod(ABC):
    def __init__(self, generation_config: GenerationConfig, sampling_config: SamplingConfig, context: SampleContext):
        self.generation_config = generation_config
        self.sampling_config = sampling_config
        self.context = context

    
    @abstractmethod
    def generate(self, datapoint: Datapoint) -> ParsedOutputGeneration | list[ParsedOutputGeneration]:
        ...




