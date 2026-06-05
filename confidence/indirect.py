import torch
from .base import ConfidenceMethod
from models.adapters import ModelScorer
from domain import ParsedOutputGeneration, ConfidenceScores, ScorerOutput

class IndirectConfidenceMethod(ConfidenceMethod):
    def __init__(self, model_scorer: ModelScorer):
        super().__init__(model_scorer)

    def tail_prompt(self, final_answer: str | int) -> str:
        return f"\nThe answer is \\boxed{{{final_answer}}}.\nTrue/False:"


    def extract(self, scorer_output: ScorerOutput) -> dict[str, float]:
        """
        ModelScorer returns a dict of {"True": logits_true, "False": logits_false} for our current sample;
        We need to softmax this to probabilities
        """
        logits = torch.tensor([scorer_output["True"], scorer_output["False"]])
        probs = torch.softmax(logits, dim=0)
        return {"True": float(probs[0]), "False": float(probs[1])}

    def compute_confidence(
        self,
        parsed_output: ParsedOutputGeneration,
    ) -> dict[str, float]:
        tail = self.tail_prompt(parsed_output.final_answer)
        if parsed_output.input_messages is not None:
            prompt = parsed_output.input_messages + [
                {"role": "assistant", "content": parsed_output.text_cot + tail}
            ]
        else:
            prompt = parsed_output.text_question + parsed_output.text_cot + tail
        scorer_output = self.model_scorer.forward_indirect(prompt, parsed_output.whole_cache)
        return self.extract(scorer_output)


