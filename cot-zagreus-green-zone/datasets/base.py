from abc import ABC, abstractmethod
from pathlib import Path

from domain import Datapoint, TrajectoryRecord, EvaluationResult, PromptRequest


class Dataset(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def load_datapoints(self) -> list[Datapoint]:
        """Load the datapoints for this benchmark NOT FROM PREGENERATED."""
        ...

    @abstractmethod
    def load_datapoints_from_pickle(self, pickle_path: str | Path) -> list[Datapoint]:
        """Load the datapoints for this benchmark FROM A PREGENERATED PICKLE."""
        ...

    @abstractmethod
    def build_messages(self, datapoint: Datapoint, prompt_request: PromptRequest) -> list[dict[str, str]]:
        ...

    @abstractmethod
    def evaluate(self, trajectory: TrajectoryRecord) -> EvaluationResult:
        ...

    @abstractmethod
    def resolve_pregenerated(self, pickle_path: str | Path) -> list[TrajectoryRecord]:
        ...




