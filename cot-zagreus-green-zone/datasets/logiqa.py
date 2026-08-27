import json
import logging
import os
import pickle
import re
import urllib.request

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import extract_mcq_letter, exact_match

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("/storage/backup/han/cot/logiqa")

_URLS = {
    "en_train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
    "en_test":  "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
    "en_eval":  "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
}

_LETTERS = "ABCD"

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "Keep your reasoning concise, and be brief in each step.\n\n"
    "When you output the final answer, output ONLY the single letter corresponding to the correct answer. Do not include periods, explanations, or any other text.\n\n"
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. Stop immediately after your last numbered reasoning step."
)

USER_PROMPT_TEMPLATE = """\
You are an expert in logical reasoning.

Context:
{context}

Question:
{question}

Options:
{options}

Reason step-by-step through the logical relationships before selecting an answer.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """


# ---------------------------------------------------------------------------
# Raw-file parser (ported from lucasmccabe/logiqa HF script)
# ---------------------------------------------------------------------------

def _process_answer(answer: str) -> str:
    if not any(answer.startswith(x) for x in "ABCD"):
        return answer
    return answer[3:]


def _process_sentences(text: str) -> str:
    text = text.replace("\n", "")
    sents = text.split(".")
    result = ""
    for sent in sents:
        if not sent:
            continue
        if not result:
            result += sent
        elif sent[0].isnumeric():
            result += "." + sent
        else:
            result += ". " + sent
    result = result.replace("  ", " ").replace("\\'", "'")
    result = result.rstrip()
    if re.match(r'^[A-Z][\w\s]+[?.!]$', result) is None:
        result += "."
    result = result.replace("?.", "?").replace("!.", "!").replace("..", ".")
    return result


def _generate_examples(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        logiqa = [_process_sentences(line) for line in f.readlines()]

    for key in range(int(len(logiqa) / 8)):
        row = 8 * key
        correct_answer = logiqa[row + 1].replace(".", "")
        context = logiqa[row + 2]
        query = logiqa[row + 3]
        answers = logiqa[row + 4 : row + 8]
        yield key, {
            "context": context,
            "query": query,
            "options": [_process_answer(answers[i]) for i in range(4)],
            "correct_option": "abcd".index(correct_answer),
        }


class LogiQADataset(Dataset):

    def __init__(self, split: str = "en_test"):
        super().__init__("logiqa")
        self.split = split

    def _download_and_parse(self) -> list[dict]:
        url = _URLS[self.split]
        filename = os.path.basename(url)
        cached_path = os.path.join(_CACHE_DIR, filename)

        if os.path.exists(cached_path):
            tmp_path = cached_path
        else:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            urllib.request.urlretrieve(url, cached_path)
            tmp_path = cached_path

        return [ex for _, ex in _generate_examples(tmp_path)]

    def load_datapoints(self) -> list[Datapoint]:
        rows = self._download_and_parse()
        entries = []
        for i, row in enumerate(rows):
            options = row["options"]
            correct_idx = row["correct_option"]
            answer_letter = _LETTERS[correct_idx]
            choices = [f"{_LETTERS[j]}. {opt}" for j, opt in enumerate(options)]

            entries.append(Datapoint(
                id=f"logiqa_{i}",
                question=row["query"],
                ground_truth=answer_letter,
                context=row["context"],
                metadata={"choices": choices},
            ))
        logger.info(f"[logiqa] {len(entries)} entries")
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"logiqa_{i}",
                question=row["question"],
                ground_truth=row["ground_truth"],
                context=row.get("context"),
                metadata=row.get("metadata", {}),
            ))
        logger.info(f"[logiqa] {len(entries)} entries loaded from {pickle_path}")
        return entries

    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        choices = datapoint.metadata.get("choices", [])
        options_text = "\n".join(choices)
        user = USER_PROMPT_TEMPLATE.format(
            context=datapoint.context or "",
            question=datapoint.question,
            options=options_text,
        )
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

        gt_stripped = str(ground_truth).strip()
        m = re.match(r"[\(\[]?\s*([A-Fa-f])\s*[\)\]]?$", gt_stripped)
        if m:
            ground_truth = m.group(1).upper()

        result = exact_match(prediction, str(ground_truth))

        trajectory.evaluation_result = EvaluationResult(
            correct=result,
            score=None,
            llm_judge_applicable=False,
        )

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError
