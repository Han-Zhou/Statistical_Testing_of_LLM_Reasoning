import logging
import pickle

from pathlib import Path

from datasets.base import Dataset
from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest
from evaluation import extract_text_answer, qa_f1_score

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
Question:
{question}

Answer the question by reasoning step-by-step.
"""

USER_PROMPT_TEMPLATE_WITH_CONTEXT = """\
Context:
{context}

Question:
{question}

Answer the question by reasoning step-by-step.
"""

ASSISTANT_PROMPT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """

F1_THRESHOLD = 0.5


class HotPotQADataset(Dataset):

    def __init__(self):
        super().__init__("hotpotqa")

    def load_datapoints(self) -> list[Datapoint]:
        try:
            import datasets as hf_datasets
        except ImportError:
            raise ImportError("Install the `datasets` package to load HotPotQA from HuggingFace.")

        ds = hf_datasets.load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
        entries = []
        for i, row in enumerate(ds):
            ctx_dict = row.get("context") or {}
            titles = ctx_dict.get("title") or []
            sent_lists = ctx_dict.get("sentences") or []
            context = "\n\n".join(
                f"{t}: {' '.join(s)}" for t, s in zip(titles, sent_lists)
            ) or None

            entries.append(Datapoint(
                id=row.get("id", f"hotpotqa_{i}"),
                question=row["question"],
                ground_truth=row["answer"],
                context=context,
            ))
        logger.info(f"[hotpotqa] {len(entries)} entries")
        return entries

    def load_datapoints_from_pickle(self, pickle_path: Path | str) -> list[Datapoint]:
        pickle_path = Path(pickle_path)
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)

        entries = []
        for i, row in enumerate(raw):
            entries.append(Datapoint(
                id=f"hotpotqa_{i}",
                question=row["question"],
                ground_truth=row["ground_truth"],
                context=row.get("context"),
            ))
        logger.info(f"[hotpotqa] {len(entries)} entries loaded from {pickle_path}")
        return entries

    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        if datapoint.context:
            user = USER_PROMPT_TEMPLATE_WITH_CONTEXT.format(
                context=datapoint.context,
                question=datapoint.question,
            )
        else:
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
                score=0.0,
                llm_judge_applicable=True,
            )
            return

        f1 = qa_f1_score(prediction, str(ground_truth))

        trajectory.evaluation_result = EvaluationResult(
            correct=f1 >= F1_THRESHOLD,
            score=f1,
            llm_judge_applicable=True,
        )

    def resolve_pregenerated(self, pickle_path):
        raise NotImplementedError
