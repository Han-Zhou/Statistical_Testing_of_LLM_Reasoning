import logging
import os
import pickle

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import extract_text_answer, normalized_text_match

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "Keep your reasoning concise, and be brief in each step.\n\n"
    "When you output the final answer, output ONLY the answer as a short string. "
    "Do not include explanations or any other text.\n\n"
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. "
    "Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. "
    "Stop immediately after your last numbered reasoning step."
)

USER_PROMPT_TEMPLATE = """\
Code snippet:
```
{code}
```

Question:
{question}

Analyze the code step-by-step and answer the question accurately.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """


class CodeQADataset(Dataset):

    def __init__(self):
        super().__init__("codeqa")

    def load_datapoints(self) -> list[Datapoint]:
        try:
            import datasets as hf_datasets
        except ImportError:
            raise ImportError("Install the `datasets` package to load CodeQA from HuggingFace.")

        local = os.environ.get("CODEQA_PATH")
        if local:
            import json
            path = Path(local)
            with open(path, "r") as f:
                try:
                    rows = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = list(hf_datasets.load_dataset("lissadesu/codeqa_v2", split="train"))

        rows = rows[2:]

        entries = []
        for i, row in enumerate(rows):
            code = row.get("code") or row.get("code_processed") or ""
            entries.append(Datapoint(
                id=str(row.get("id", i)),
                question=row["question"],
                ground_truth=row.get("answer", ""),
                context=code or None,
            ))
        logger.info(f"[codeqa] {len(entries)} entries")
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"codeqa_{i}",
                question=row["question"],
                ground_truth=row["ground_truth"],
                context=row.get("context"),
            ))
        logger.info(f"[codeqa] {len(entries)} entries loaded from {pickle_path}")
        return entries

    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        code = datapoint.context or ""
        user = USER_PROMPT_TEMPLATE.format(code=code, question=datapoint.question)
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
                llm_judge_applicable=True,
            )
            return

        result = normalized_text_match(prediction, str(ground_truth))

        trajectory.evaluation_result = EvaluationResult(
            correct=result,
            score=None,
            llm_judge_applicable=True,
        )

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError
