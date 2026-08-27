from models.core_models.api_llm import API_LLM
from models.adapters.gpt_adapter import GptAdapter, GptScorer


class QwenAscendScorer(GptScorer):
    """Inherits all scoring logic from GptScorer unchanged.

    The Ascend gateway returns logprobs in the same OpenAI format. API_LLM.forward()
    disables thinking via chat_template_kwargs.enable_thinking=False so the model
    emits the answer token directly, making logprobs.content[0] the answer
    distribution.
    """
    pass


class QwenAscendAdapter(GptAdapter):
    def __init__(self):
        self.model = API_LLM(model_name="qwen_ascend")
        self.model_scorer = QwenAscendScorer(self.model)

    def generate_helper(self, prompt, max_tokens, cache, temperature):
        # Phase 1: the Ascend-served Qwen is a reasoning model. The CoT lives in
        # message.reasoning_content; message.content holds only the bare answer,
        # which we discard (phase 2 produces the boxed answer with real logprobs).
        phase_1 = self.model.generate(
            prompt_messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort="medium",
        )
        message = phase_1.outputs.choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or ""
        post_thinking_content = getattr(message, "content", None) or ""
        phase_1_cot = self._strip_trailing_bare_answer(post_thinking_content)

        prefix = "\nThe answer is \\boxed{"
        phase_2_messages = list(prompt) + [
            {"role": "assistant", "content": phase_1_cot + prefix}
        ]
        phase_2 = self.model.generate(
            prompt_messages=phase_2_messages,
            max_tokens=16,
            temperature=temperature,
            stop=["}"],
            continue_final_message=True,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )

        return self._merge_two_phase(prompt, phase_1_cot, prefix, phase_2)
