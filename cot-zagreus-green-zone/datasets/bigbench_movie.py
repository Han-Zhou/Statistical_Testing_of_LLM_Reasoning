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
    "When you output the final answer, output ONLY the single letter corresponding to the correct answer. Do not include periods, explanations, or any other text.\n\n"
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. Stop immediately after your last numbered reasoning step."
)



USER_PROMPT_TEMPLATE = """\
{question}

Think carefully about the movies listed and the options provided. Reason \
step-by-step before choosing the best answer.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """



class BigBenchMovieDataset(Dataset):

    def __init__(self):
        super().__init__("bigbench_movie")


    def _fetch_bbh(self, config: str) -> list[dict]:
        """Download a BBH (BigBench Hard) config JSON and return the examples list."""
        _BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{}.json"
        url = _BBH_URL.format(config)
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())["examples"]
        


    def load_datapoints(self) -> list[Datapoint]:
        """Loads the full BBH Movie Recommendation dataset as a list of Datapoints."""
        rows = self._fetch_bbh("movie_recommendation")
        entries = []
        for i, row in enumerate(rows):
            entries.append(Datapoint(
                id=f"bigbench_movie_{i}",
                question=row["input"],
                ground_truth=row["target"],
            ))
        logger.info(f"[bigbench_movie] {len(entries)} entries")
        return entries


    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"bigbench_movie_{i}",
                # id_original=row["id"],
                question=row["question"],
                ground_truth=row["ground_truth"],
            ))
        logger.info(f"[bigbench_movie] {len(entries)} entries loaded from {pickle_path}")
        return entries


    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        user = USER_PROMPT_TEMPLATE.format(question=datapoint.question)
        assistant = ASSISTANT_PROMPT_START

        if prompt_request.prompt_type == 1:
            # Type 1: assistant prefill with thinking tokens and 'Step 1:'
            # Currently neither llama nor qwen support this prompt type (qwen does assistant prefill in the non-thinking phase)
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        elif prompt_request.prompt_type == 2:
            # Type 2: 'Let's think step-by-step.' in user prompt, no assistant prefill.
            # currently both llama and qwen support this prompt type
            user += "\nLet's think step-by-step."
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        

    def evaluate(self, trajectory: TrajectoryRecord) -> None:
        """
        mutates the trajectory record by filling in the evaluation_result field, which includes correctness and other evaluation metadata.
        """
        ground_truth = trajectory.ground_truth
        prediction = trajectory.final_answer

        if prediction is None:
            trajectory.evaluation_result = EvaluationResult(
                correct=False,
                score=None,
                llm_judge_applicable=False,
            )
            return

        # Normalize MCQ ground truth: strip surrounding parens/brackets so "(C)" -> "C" to match the extractor output format
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


