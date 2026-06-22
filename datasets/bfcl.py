import json
import logging
import pickle

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import normalized_text_match

logger = logging.getLogger(__name__)

_BFCL_DIR = Path("/storage/backup/han/cot/bfcl")
_TYPES = ("simple_python", "multiple", "parallel", "parallel_multiple")

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "Keep your reasoning concise, and be brief in each step.\n\n"
    "When you output the final answer, output ONLY the function call(s) in the format "
    "[func_name(param=value, ...)] with no other text.\n\n"
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. "
    "Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. "
    "Stop immediately after your last numbered reasoning step."
)

USER_PROMPT_TEMPLATE = """\
You are also an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your FINAL response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]. You SHOULD NOT include any other text in the FINAL response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{functions}

Question:
{question}
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """


def _load_json_or_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


class BfclDataset(Dataset):

    def __init__(self):
        super().__init__("bfcl")

    def load_datapoints(self) -> list[Datapoint]:
        all_entries: list[Datapoint] = []
        for bfcl_type in _TYPES:
            entries = self._load_type(bfcl_type)
            logger.info(f"  [bfcl/{bfcl_type}] {len(entries)} entries")
            all_entries.extend(entries)
        logger.info(f"[bfcl] {len(all_entries)} total entries")
        return all_entries

    def _load_type(self, bfcl_type: str) -> list[Datapoint]:
        if bfcl_type == "parallel_multiple":
            q_path = _BFCL_DIR / "v4_parallel_multiple_answer.json"
            a_path = _BFCL_DIR / "v4_parallel_multiple.json"
        else:
            q_path = _BFCL_DIR / f"v4_{bfcl_type}.json"
            a_path = _BFCL_DIR / f"v4_{bfcl_type}_answer.json"

        questions = _load_json_or_jsonl(q_path)
        answers = {
            row["id"]: row["ground_truth"]
            for row in _load_json_or_jsonl(a_path)
        }

        entries = []
        for q in questions:
            qid = q["id"]
            turns = q["question"][0] if q["question"] else []
            user_msg = next(
                (t["content"] for t in turns if t.get("role") == "user"), ""
            )
            ground_truth = answers.get(qid, [])
            entries.append(Datapoint(
                id=qid,
                question=user_msg,
                ground_truth=json.dumps(ground_truth),
                metadata={
                    "functions": q.get("function", []),
                    "bfcl_type": bfcl_type,
                },
            ))
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=row.get("id", f"bfcl_{i}"),
                question=row["question"],
                ground_truth=row["ground_truth"],
                metadata=row.get("metadata", {}),
            ))
        logger.info(f"[bfcl] {len(entries)} entries loaded from {pickle_path}")
        return entries

    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        functions = datapoint.metadata.get("functions", [])
        functions_str = json.dumps(functions, indent=2)
        user = USER_PROMPT_TEMPLATE.format(
            question=datapoint.question,
            functions=functions_str,
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
        prediction = trajectory.final_answer

        if prediction is None:
            trajectory.evaluation_result = EvaluationResult(
                correct=False,
                score=None,
                llm_judge_applicable=True,
            )
            return

        result = normalized_text_match(prediction, str(trajectory.ground_truth))

        trajectory.evaluation_result = EvaluationResult(
            correct=result,
            score=None,
            llm_judge_applicable=True,
        )

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError
