import re
import numpy as np

from pipeline.sampling.base import SamplingMethod

from config import GenerationConfig, SamplingConfig
from models.adapters.base import ModelAdapter
from domain.data import Datapoint, ParsedOutputGeneration, PromptRequest

from pipeline.sampling.context import SampleContext



_STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)

def _resample_steps(steps: list[str], rng: np.random.Generator) -> list[str]:
    if len(steps) < 2:
        return list(steps)
    prefix = steps[:-1]
    resampled = list(rng.choice(prefix, size=len(prefix), replace=True))
    return resampled + [steps[-1]]

def _rebuild_cot(steps: list[str]) -> str:
    # Renumber 'Step N:' so the result is monotonic; leave non-marker steps as-is.
    out, n = [], 1
    for s in steps:
        m = _STEP_MARKER_RE.match(s)
        if m:
            s = _STEP_MARKER_RE.sub(f"Step {n}:", s, count=1)
            n += 1
        out.append(s)
    return "\n\n".join(out)
    
    
    
class StepBootstrapSampling(SamplingMethod):
    def __init__(self, generation_config: GenerationConfig, sampling_config: SamplingConfig, context: SampleContext):
        super().__init__(generation_config, sampling_config, context)


    def _alternative_cots_stepbootstrap(self) -> list[str]:
        rng = np.random.default_rng(self.sampling_config.seed_stepbootstrap)
        return [_rebuild_cot(_resample_steps(self.context.reference_vanilla_cot, rng)) 
        for _ in range(self.sampling_config.nb_stepbootstrap_samples)]



    def _add_assistant_message_to_messages(self, messages: list[dict[str, str]], alternative_cot: str) -> list[dict[str, str]]:
        """
        adds the alternative_cots to the messages
        also:
        - if hf backend, adds the answer to the messages
        - if api backend, do NOT add the answer to the messages
        Does NOT mutatate the original messages
        """
        new_messages = [message.copy() for message in messages]
        if self.generation_config.backend == "hf":
            final_answer_sentence = f"\nTherefore the final answer is \\boxed{{{self.context.reference_vanilla_final_answer}}}."
        else:
            final_answer_sentence = f"\nTherefore the final answer is \\boxed{{"
        if self.generation_config.prompt_type == 1:
            # we append to the end  - the assistant prefill is already there
            new_messages[-1]["content"] += f"\n{alternative_cot}{final_answer_sentence}"
        else:
            assistant_message = {"role": "assistant", "content": alternative_cot + final_answer_sentence}
            new_messages.append(assistant_message)

        return new_messages



    def generate(self) -> list[ParsedOutputGeneration]:
        """
        For StepBootstrap sampling, it is more of a forward pass.
        We use:
        - the original prompt
        - resampled cot steps with replacement, and indexed correctly
        - the original final answer
        """

        messages = self.context.dataset.build_messages(
            self.context.datapoint,
            prompt_request=PromptRequest(few_shot=False, prompt_type=self.generation_config.prompt_type),
        )

        alternative_cots = self._alternative_cots_stepbootstrap()

        generation_outputs: list[ParsedOutputGeneration] = []

        # NOTE possible optimization for vllm backend 

        for i in range(self.sampling_config.nb_stepbootstrap_samples):
            new_messages = self._add_assistant_message_to_messages(messages, alternative_cots[i])

            generate_output: ParsedOutputGeneration = self.context.model_adapter.forward_pass(
                messages=new_messages,
                cache=self.context.reference_vanilla_question_cache,
            )

            # if api backend, it is possible that the answer token ids are not the same as the reference answer token ids
            # in this case, we need to extract the answer token ids from the generate output
            if self.generation_config.backend == "api":
                generate_output.answer_token_ids = self.context.reference_vanilla_answer_tokens_for_api

            generation_outputs.append(generate_output)

        return generation_outputs
