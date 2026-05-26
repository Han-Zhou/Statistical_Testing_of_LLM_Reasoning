import logging

from transformers.cache_utils import DynamicCache

from pipeline.sampling.base import SamplingMethod

from config import GenerationConfig, SamplingConfig
from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration, PromptRequest, KVCache, CacheBundle

from pipeline.sampling.context import SampleContext


logger = logging.getLogger(__name__)


class RejectionSampling(SamplingMethod):
    def __init__(self, generation_config: GenerationConfig, sampling_config: SamplingConfig, context: SampleContext):
        super().__init__(generation_config, sampling_config, context)


   

    def generate(self) -> list[ParsedOutputGeneration]:
        """
        For Rejection sampling, we generate nb_cot_samples samples with temperature != 0.0,
        and we reject the ones that are not correct according to the confidence engine.
        """
        # few_shot not implemented yet
        messages = self.context.dataset.build_messages(
            self.context.datapoint,
            prompt_request=PromptRequest(few_shot=False, prompt_type=self.generation_config.prompt_type),
        )

        generation_outputs: list[ParsedOutputGeneration] = []
        for _ in range(self.sampling_config.nb_cot_samples):
            generate_output: ParsedOutputGeneration = self.context.model_adapter.generate(
                messages=messages,
                max_tokens=self.generation_config.max_tokens,
                cache=self.context.reference_vanilla_question_cache,
                temperature=self.sampling_config.temperature,
            )

            generation_outputs.append(generate_output)

        return generation_outputs
