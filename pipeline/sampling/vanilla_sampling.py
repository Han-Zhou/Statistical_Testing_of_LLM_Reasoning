

from pipeline.sampling.base import SamplingMethod

from config import GenerationConfig, SamplingConfig
from pipeline.sampling.context import SampleContext
from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration, PromptRequest, KVCache


class VanillaSampling(SamplingMethod):
    def __init__(self, generation_config: GenerationConfig, sampling_config: SamplingConfig, context: SampleContext):
        super().__init__(generation_config, sampling_config, context)
        
    def generate(self) -> ParsedOutputGeneration:
        """
        For vanilla sampling, we just do one generation pass and parse the output, with temperature = 0.0.
        """
        # few_shot not implemented yet
        messages = self.context.dataset.build_messages(self.context.datapoint, prompt_request=PromptRequest(few_shot=False, prompt_type=self.generation_config.prompt_type))

        generate_output = self.context.model_adapter.generate(
            messages=messages,
            max_tokens=self.generation_config.max_tokens,
        )

        return generate_output










