import torch
from .base import ConfidenceMethod
from domain import ParsedOutputGeneration, ConfidenceScores, ScorerOutput
from models.adapters import ModelScorer


class VerbalConfidenceMethod(ConfidenceMethod):
    def __init__(self, model_scorer: ModelScorer):
        super().__init__(model_scorer)
    
    def tail_prompt(self, final_answer: str | int) -> str:
        return (
            f"\nThe answer is \\boxed{{{final_answer}}}.\n"
            f"Please respond with a score from 0 to 100 in <confidence> </confidence> tags.\n"
            f"How confident are you in your previous answer?\n"
            f"<confidence>"
        )

    def extract(self, scorer_output: ScorerOutput) -> dict[str, float]:
        """
        ModelScorer returns a dict of {int (in string form): logits_score} for our current sample;
        We need to convert this to softmaxed probabilities
        """
        logits = torch.tensor([scorer_output[score] for score in scorer_output.keys()])
        probs = torch.softmax(logits, dim=0).detach().cpu()
        return {int(int(score)): float(probs[i]) for i, score in enumerate(scorer_output.keys())}

    def compute_confidence(
        self,
        parsed_output: ParsedOutputGeneration,
    ) -> dict[str, float]:
        prompt = parsed_output.text_question + parsed_output.text_cot + self.tail_prompt(parsed_output.final_answer)
        scorer_output = self.model_scorer.forward_verbal(prompt, parsed_output.whole_cache)
        return self.extract(scorer_output)


