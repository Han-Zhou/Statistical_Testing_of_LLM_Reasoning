from pipeline.sampling.base import SamplingMethod

from config import GenerationConfig, SamplingConfig
from models.adapters.base import ModelAdapter
from domain import Datapoint, ParsedOutputGeneration, PromptRequest
from pipeline.sampling.context import SampleContext



class LawyerSampling(SamplingMethod):
    def __init__(self, generation_config: GenerationConfig, sampling_config: SamplingConfig, context: SampleContext):
        super().__init__(generation_config, sampling_config, context)

    def _build_messages(self) -> list[dict[str, str]]:
        messages = self.context.dataset.build_messages(
            self.context.datapoint,
            prompt_request=PromptRequest(few_shot=False, prompt_type=self.generation_config.prompt_type),
        )
        if self.generation_config.prompt_type == 1:
            # type 1: append at end of user message
            messages[-2]["content"] += f"\nExplain why the final answer is \\boxed{{{self.context.reference_vanilla_final_answer}}}."
        else:
            # type 2: insert explanation request before "\nLet's think step-by-step." in the user message
            to_insert = f"\nExplain why the final answer is \\boxed{{{self.context.reference_vanilla_final_answer}}}."
            user_content = messages[-1]["content"]
            marker = "\nLet's think step-by-step." # guaranteed to be in the user message
            idx = user_content.index(marker)
            messages[-1]["content"] = user_content[:idx] + to_insert + user_content[idx:]
     
        return messages

    def generate(self) -> list[ParsedOutputGeneration]:
        """
        For Lawyer sampling, we:
        - add a string at end of prompt, asking the model to defend the final answer 
        - generate nb_cot_samples samples with temperature != 0.0,
        """
        # few_shot not implemented yet
        messages = self._build_messages()

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



        