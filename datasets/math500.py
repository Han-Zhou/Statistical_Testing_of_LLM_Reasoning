import logging
import pickle

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import extract_boxed_or_text, math_equal

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "Keep your reasoning concise, and be brief in each step.\n\n"
    "When you output the final answer, output ONLY the exact numerical or symbolic answer. "
    "Do not include units or explanations.\n\n"
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. "
    "Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. "
    "Stop immediately after your last numbered reasoning step."
)

USER_PROMPT_TEMPLATE = """\
Question: {question}

Solve step-by-step, showing all reasoning clearly.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """


class Math500Dataset(Dataset):

    def __init__(self):
        super().__init__("math500")

    def load_datapoints(self) -> list[Datapoint]:
        try:
            import datasets as hf_datasets
        except ImportError:
            raise ImportError("Install the `datasets` package to load MATH-500 from HuggingFace.")

        ds = hf_datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        entries = []
        for i, row in enumerate(ds):
            if i < 2:
                continue
            entries.append(Datapoint(
                id=str(row.get("unique_id", i)),
                question=row["problem"],
                ground_truth=row.get("answer", row.get("solution", "")),
            ))
        logger.info(f"[math500] {len(entries)} entries")
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"math500_{i}",
                question=row["question"],
                ground_truth=row["ground_truth"],
            ))
        logger.info(f"[math500] {len(entries)} entries loaded from {pickle_path}")
        return entries

    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        user = USER_PROMPT_TEMPLATE.format(question=datapoint.question)
        assistant = ASSISTANT_PROMPT_START

        if prompt_request.prompt_type == 1:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        elif prompt_request.prompt_type == 2:
            user += "\nLet's think step-by-step."
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

    def evaluate(self, trajectory: TrajectoryRecord) -> None:
        ground_truth = trajectory.ground_truth
        prediction = trajectory.final_answer

        if prediction is None:
            trajectory.evaluation_result = EvaluationResult(
                correct=False,
                score=None,
                llm_judge_applicable=False,
            )
            return

        result = math_equal(prediction, str(ground_truth))

        trajectory.evaluation_result = EvaluationResult(
            correct=result,
            score=None,
            llm_judge_applicable=False,
        )

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError
