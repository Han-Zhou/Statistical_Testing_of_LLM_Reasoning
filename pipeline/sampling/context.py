
from dataclasses import dataclass

from datasets import Dataset
from models.adapters import ModelAdapter


@dataclass
class SampleContext:
    model_adapter: ModelAdapter
    dataset: Dataset
