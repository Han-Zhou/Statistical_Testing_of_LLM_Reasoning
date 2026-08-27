import json
import logging
import pickle

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import extract_text_answer, normalized_text_match

logger = logging.getLogger(__name__)

_CS1QA_DIR = Path("/storage/backup/han/cot/cs1qa")

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
Code:
```
{code}
```

Question:
{question}

Analyze the code step-by-step and answer the question clearly and concisely.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """


def _dedup_key(row: dict) -> tuple:
    return (
        row.get("labNo"),
        row.get("taskNo"),
        row.get("questioner"),
        row.get("question"),
        row.get("code"),
    )


def _load_and_group(path: Path) -> dict[tuple, list[dict]]:
    with open(path, encoding="utf-8") as f:
        try:
            rows = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            rows = [json.loads(line) for line in f if line.strip()]

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = _dedup_key(row)
        groups.setdefault(key, []).append(row)
    return groups


class CS1QADataset(Dataset):

    def __init__(self):
        super().__init__("cs1qa")

    def load_datapoints(self) -> list[Datapoint]:
        path = _CS1QA_DIR / "test_cleaned.jsonl"
        groups = _load_and_group(path)
        entries = []
        for i, (key, rows) in enumerate(groups.items()):
            rep = rows[0]
            answer = rep.get("answer", "")
            if not answer:
                continue
            entries.append(Datapoint(
                id=f"cs1qa_{i}",
                question=rep["question"],
                ground_truth=str(answer),
                context=rep.get("code"),
            ))
        logger.info(f"[cs1qa] {len(entries)} entries")
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"cs1qa_{i}",
                question=row["question"],
                ground_truth=row["ground_truth"],
                context=row.get("context"),
            ))
        logger.info(f"[cs1qa] {len(entries)} entries loaded from {pickle_path}")
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
