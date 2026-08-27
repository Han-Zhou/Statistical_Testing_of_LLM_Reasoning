import json
import logging
import pickle
import urllib.request
import re

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import extract_mcq_letter, exact_match

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "Keep your reasoning concise, and be brief in each step.\n\n"
    "When you output the final answer, output ONLY 'Yes' or 'No'. "
    "Do not include periods, explanations, or any other text.\n\n"
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. "
    "Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. "
    "Stop immediately after your last numbered reasoning step."
)

USER_PROMPT_TEMPLATE = """\
{question}

Reason through the causal relationships step-by-step before answering.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """


class BigBenchCausalDataset(Dataset):

    def __init__(self):
        super().__init__("bigbench_causal")

    def _fetch_bbh(self, config: str) -> list[dict]:
        _BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{}.json"
        url = _BBH_URL.format(config)
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())["examples"]

    def load_datapoints(self) -> list[Datapoint]:
        rows = self._fetch_bbh("causal_judgement")
        entries = []
        for i, row in enumerate(rows):
            entries.append(Datapoint(
                id=f"bigbench_causal_{i}",
                question=row["input"],
                ground_truth=row["target"],
            ))
        logger.info(f"[bigbench_causal] {len(entries)} entries")
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"bigbench_causal_{i}",
                question=row["question"],
                ground_truth=row["ground_truth"],
            ))
        logger.info(f"[bigbench_causal] {len(entries)} entries loaded from {pickle_path}")
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

        result = exact_match(prediction, str(ground_truth))

        trajectory.evaluation_result = EvaluationResult(
            correct=result,
            score=None,
            llm_judge_applicable=False,
        )

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError
