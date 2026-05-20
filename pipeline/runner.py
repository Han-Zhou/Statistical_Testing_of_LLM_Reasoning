
import time

from config import GenerationConfig, ConfidenceConfig, SamplingConfig
from confidence import ConfidenceEngine
# from models import LLM, API_LLM
from models.adapters import ModelAdapter
from models.registry import MODEL_ADAPTER_REGISTRY

from domain.data import ParsedOutputGeneration, Datapoint, TrajectoryRecord
from pipeline.sampling import SamplingMethod, SamplingMethodFactory, SampleContext

from confidence.confidence_engine import ConfidenceEngine 

from repository import TrajectoryRepository
from datasets import DATASETS, Dataset



class Runner:
    def __init__(
        self,
        generation_config: GenerationConfig,
        confidence_config: ConfidenceConfig,
        sampling_config: SamplingConfig,
    ):
        self.generation_config = generation_config
        self.confidence_config = confidence_config
        self.sampling_config = sampling_config

        

        self.confidence_engine: ConfidenceEngine = ConfidenceEngine(self.confidence_config)

        self.model_adapter = MODEL_ADAPTER_REGISTRY[self.generation_config.model]()

        # Datasets
        self.dataset: Dataset = DATASETS[self.generation_config.dataset]()

        self.context = SampleContext(model_adapter=self.model_adapter, dataset=self.dataset)

        # Generation and confidence
        (
            self.vanilla_sampling,
            self.rejection_sampling,
            self.lawyer_sampling,
            self.stepbootstrap_sampling
        ) = SamplingMethodFactory.create(
            generation_config=self.generation_config,
            sampling_config=self.sampling_config,
            context=self.context
        )

        # Repository
        self.trajectory_repository = TrajectoryRepository("trajectories/trial_520")

        
    def run_generation_and_confidence(self, datapoint: Datapoint):
        # 1 - generate vanilla samples & confidences
        T0 = time.perf_counter()
        vanilla_generation_output: ParsedOutputGeneration = self.vanilla_sampling.generate(datapoint=datapoint)
        T1 = time.perf_counter()
        # vanilla_confidence = self.confidence_engine.compute_confidence(vanilla_generation_output)
        T2 = time.perf_counter()
        vanilla_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

        # # 2 - generate rejection samples & confidences
        # T0 = time.perf_counter()
        # rejection_generation_outputs: list[ParsedOutputGeneration] = self.rejection_sampling.generate(datapoint=datapoint)
        # T1 = time.perf_counter()
        # rejection_confidences = [self.confidence_engine.compute_confidence(output) for output in rejection_generation_outputs]
        # T2 = time.perf_counter()
        # rejection_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

        # # 3 - generate lawyer samples & confidences
        # T0 = time.perf_counter()
        # lawyer_generation_outputs: list[ParsedOutputGeneration] = self.lawyer_sampling.generate(datapoint=datapoint)
        # T1 = time.perf_counter()
        # lawyer_confidences = [self.confidence_engine.compute_confidence(output) for output in lawyer_generation_outputs]
        # T2 = time.perf_counter()
        # lawyer_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

        # # 4 - generate stepbootstrap samples & confidences
        # T0 = time.perf_counter()
        # stepbootstrap_generation_outputs: list[ParsedOutputGeneration] = self.stepbootstrap_sampling.generate(datapoint=datapoint)
        # T1 = time.perf_counter()
        # stepbootstrap_confidences = [self.confidence_engine.compute_confidence(output) for output in stepbootstrap_generation_outputs] 
        # T2 = time.perf_counter()
        # stepbootstrap_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

        # NOTE output the results
        self.trajectory_repository.save(
            trajectory_record=TrajectoryRecord(
                id=datapoint.id,
                question=datapoint.question,
                ground_trtuth=datapoint.ground_truth,
                prompt=vanilla_generation_output.text_question,
                generated_text=vanilla_generation_output.text_cot_with_answer,
                cot_steps=vanilla_generation_output.cot_steps,
                final_answer=vanilla_generation_output.final_answer,
                # NOTE correctness / evaluation not implemented yet
                correct=None,
                confidences=None,
                confidence_timings=None
            )
        )


    def run(self):
        if self.generation_config.from_pickle is not None:
            datapoints: list[Datapoint] = self.dataset.load_datapoints_from_pickle(self.generation_config.from_pickle)

            for datapoint in datapoints:
                self.run_generation_and_confidence(datapoint)
        
        

        



