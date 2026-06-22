"""
Trajectory Repository is the bridge between the trajectory data and the rest of the application. 
"""

import json
import logging
import traceback
from typing import Any
from dataclasses import asdict, fields, replace
from pathlib import Path

from domain import TrajectoryRecord, ConfidenceScores, Timings, EvaluationResult

logger = logging.getLogger(__name__)


class TrajectoryRepository:
    def __init__(self, trajectory_data_path: Path | str):
        self.trajectory_data_path = Path(trajectory_data_path) if isinstance(trajectory_data_path, str) else trajectory_data_path
        self.trajectory_data_path.mkdir(parents=True, exist_ok=True)

    # TODO: update loading logic
    def load(self, index: int, sample: int | None = None) -> TrajectoryRecord:
        if sample is not None:
            file_name = f"traj_{index}_sample_{sample}.json"
        else:
            file_name = f"traj_{index}.json"
        
        file_path = self.trajectory_data_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")
    
        with open(file_path, "r") as f:
            data = json.load(f)

        evaluation_result = data.get("evaluation_result")
        if evaluation_result is not None:
            evaluation_result = EvaluationResult(**evaluation_result)

        confidences = data.get("confidences")
        if confidences is not None:
            confidences = ConfidenceScores(**confidences)

        timings = data.get("timings")
        if timings is not None:
            timings = Timings(**timings)

        return TrajectoryRecord(
            id=data["id"],
            question=data["question"],
            prompt=data["prompt"],
            generated_text=data.get("generated_text", None),
            cot_steps=data.get("cot_steps", []),
            final_answer=data.get("final_answer", None),
            ground_truth=data["ground_truth"],
            evaluation_result=evaluation_result,
            #  NOTE: not sure if we are even saving prompt_cache
            # prompt_cache_path=data.get("prompt_cache_path", None)
            # NOTE: confidence + confidence_timing loading slightly hacky
            confidences=confidences,
            timings=timings,
            input_messages=data.get("input_messages", None),
        )


    def save(self, trajectory_record: TrajectoryRecord, sample: int | None = None):
        # Extract rightmost int from trajectory_record.id
        # Should be fine since all of our ids are in the format {dataset}_i
        id_parts = str(trajectory_record.id).split('_')
        record_id = id_parts[-1] if id_parts[-1].isdigit() else trajectory_record.id
        
        if sample is not None:
            file_name = f"traj_{record_id}_sample_{sample}.json"
        else:
            file_name = f"traj_{record_id}.json"
        
        file_path = self.trajectory_data_path / file_name
        
        if file_path.exists():
            logger.warning(f"Trajectory file already exists and will be overwritten: {file_path}")

        
        with open(file_path, "w") as f:
            json.dump({
                "id": trajectory_record.id,
                "question": trajectory_record.question,
                "prompt": trajectory_record.prompt,
                "generated_text": trajectory_record.generated_text,
                "cot_steps": trajectory_record.cot_steps,
                "final_answer": trajectory_record.final_answer,
                "ground_truth": trajectory_record.ground_truth,
                "evaluation_result": asdict(trajectory_record.evaluation_result) if trajectory_record.evaluation_result is not None else None,
                # "prompt_cache_path": trajectory_record.prompt_cache_path,
                "confidences": asdict(trajectory_record.confidences) if trajectory_record.confidences is not None else None,
                "timings": asdict(trajectory_record.timings) if trajectory_record.timings is not None else None,
                "input_messages": trajectory_record.input_messages,
                "cost (cumulative)": trajectory_record.cost,
            }, f, indent=2)
        


    def add(self, index: int, field_name: str, value: Any, sample: int | None = None):
        """Add/overwrite a single field in an existing trajectory JSON. Debugging only."""
        file_name = f"traj_{index}_sample_{sample}.json" if sample is not None else f"traj_{index}.json"
        file_path = self.trajectory_data_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        data[field_name] = value

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


    def update(self, trajectory_record: TrajectoryRecord, fields_to_update: list[str], sample: int | None = None):
        existing_record = self.load(trajectory_record.id, sample)
        if existing_record is None:
            raise ValueError(f"Trajectory record with id {trajectory_record.id} does not exist for appending.")

        valid_fields = {f.name for f in fields(TrajectoryRecord)}
        unknown = set(fields_to_update) - valid_fields
        if unknown:
            raise ValueError(f"Unknown fields requested for update: {sorted(unknown)}")

        updates = {name: getattr(trajectory_record, name) for name in fields_to_update}
        updated_record = replace(existing_record, **updates)

        self.save(updated_record, sample)


    def save_error(self, datapoint_id, error: BaseException, sample: int | None = None):
        id_parts = str(datapoint_id).split('_')
        record_id = id_parts[-1] if id_parts[-1].isdigit() else datapoint_id

        if sample is not None:
            file_name = f"traj_{record_id}_sample_{sample}_error.json"
        else:
            file_name = f"traj_{record_id}_error.json"

        file_path = self.trajectory_data_path / file_name
        with open(file_path, "w") as f:
            json.dump({
                "id": datapoint_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exception(type(error), error, error.__traceback__),
            }, f, indent=2)
        

