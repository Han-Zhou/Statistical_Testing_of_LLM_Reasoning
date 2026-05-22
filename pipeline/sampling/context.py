
from dataclasses import dataclass

from datasets import Dataset
from models.adapters import ModelAdapter
from domain.data import CachedPrefix, Datapoint

@dataclass
class SampleContext:
    model_adapter: ModelAdapter
    dataset: Dataset
    datapoint: Datapoint | None = None
    reference_vanilla_cot: list[str] | None = None
    reference_vanilla_final_answer: str | None = None
    reference_vanilla_question_prefix: CachedPrefix | None = None

    def clear(self):
        for field_name in list(self.__dataclass_fields__):
            if field_name.startswith("reference_"):
                setattr(self, field_name, None)
        setattr(self, "datapoint", None)

