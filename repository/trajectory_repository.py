"""
Trajectory Repository is the bridge between the trajectory data and the rest of the application. 
"""

import json
import logging
from dataclasses import fields, replace
from pathlib import Path

from domain.data import TrajectoryRecord

logger = logging.getLogger(__name__)


class TrajectoryRepository:
    def __init__(self, trajectory_data_path: Path):
        self.trajectory_data_path = trajectory_data_path

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
        
        return TrajectoryRecord(
            id=data["id"],
            question=data["question"],
            prompt=data["prompt"],
            generated_text=data.get("generated_text", None),
            cot_steps=data.get("cot_steps", []),
            final_answer=data.get("final_answer", None),
            ground_truth=data["ground_truth"],
            correct=data.get("correct", None),
            #  NOTE: not sure if we are even saving prompt_cache
            # prompt_cache_path=data.get("prompt_cache_path", None)
            # NOTE: confidence loading slightly hacky
            confidences=data.get("confidences", None) 
        )


    def save(self, trajectory_record: TrajectoryRecord, sample: int | None = None):
        if sample is not None:
            file_name = f"traj_{trajectory_record.id}_sample_{sample}.json"
        else:
            file_name = f"traj_{trajectory_record.id}.json"
        
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
                "correct": trajectory_record.correct,
                # "prompt_cache_path": trajectory_record.prompt_cache_path,
                "confidences": trajectory_record.confidences
            }, f, indent=2)
        




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
        

