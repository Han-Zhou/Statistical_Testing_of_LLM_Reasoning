
import time

from config import GenerationConfig, ConfidenceConfig, SamplingConfig
from confidence import ConfidenceEngine
# from models import LLM, API_LLM
from models.adapters import ModelAdapter
from models.registry import MODEL_ADAPTER_REGISTRY

from domain import ParsedOutputGeneration, Datapoint, TrajectoryRecord, EvaluationResult, ConfidenceScores, Timings
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

        self.model_adapter = MODEL_ADAPTER_REGISTRY[self.generation_config.model]()

        self.confidence_engine: ConfidenceEngine = ConfidenceEngine(self.confidence_config, self.model_adapter.model_scorer)

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
        ss = self.generation_config.sample_size if self.generation_config.sample_size is not None else None
        sr = f"{self.generation_config.sample_range[0]}_{self.generation_config.sample_range[1]}" if self.generation_config.sample_range is not None else None
        # samples is either ss or sr but not both. If both are None, sample_range is "full".
        if ss is not None:
            samples = ss
        elif sr is not None:
            samples = sr
        else:
            samples = "full"
        base_dir_name = f"trajectories/{self.generation_config.tag}_{self.generation_config.model}_{self.generation_config.dataset}_s{samples}"
        self.vanilla_trajectory_repository = TrajectoryRepository(f"{base_dir_name}/vanilla")
        self.rejection_trajectory_repository = TrajectoryRepository(f"{base_dir_name}/rejection")
        self.lawyer_trajectory_repository = TrajectoryRepository(f"{base_dir_name}/lawyer")
        self.stepbootstrap_trajectory_repository = TrajectoryRepository(f"{base_dir_name}/stepbootstrap")


    def _run_generation_and_confidence_vanilla(self):
        """
        - generate vanilla samples & confidences;
        - outputs to the vanilla dir
        - mutate the reference_* fields in self.context
        """
        # generate vanilla samples & confidences
        datapoint = self.context.datapoint
        T0 = time.perf_counter()
        vanilla_generation_output: ParsedOutputGeneration = self.vanilla_sampling.generate()
        T1 = time.perf_counter()
        vanilla_confidence = self.confidence_engine.compute_confidence(vanilla_generation_output)
        T2 = time.perf_counter()

        # update context
        self.context.reference_vanilla_cot = vanilla_generation_output.cot_steps
        self.context.reference_vanilla_final_answer = vanilla_generation_output.final_answer
        self.context.reference_vanilla_question_cache = vanilla_generation_output.question_cache

        record = TrajectoryRecord(
                    id=datapoint.id,
                    question=datapoint.question,
                    ground_truth=datapoint.ground_truth,
                    prompt=vanilla_generation_output.text_question,
                    generated_text=vanilla_generation_output.text_cot_with_answer,
                    cot_steps=vanilla_generation_output.cot_steps,
                    final_answer=vanilla_generation_output.final_answer,
                    evaluation_result=None,
                    confidences=vanilla_confidence,
                    timings=Timings(
                        generation_time=T1-T0,
                        confidence_time=T2-T1,
                    )
                )
        
        self.dataset.evaluate(record)
        
        # save vanilla trajectory
        self.vanilla_trajectory_repository.save(
            trajectory_record=record
        )


    def _run_generation_and_confidence_rejection(self):
        """
        - generate rejection samples & confidences;
        - outputs to the rejection dir (one file per sample)
        """
        # generate rejection samples & confidences
        datapoint = self.context.datapoint
        T0 = time.perf_counter()
        rejection_generation_outputs: list[ParsedOutputGeneration] = self.rejection_sampling.generate()
        T1 = time.perf_counter()
        rejection_confidences = [self.confidence_engine.compute_confidence(output) for output in rejection_generation_outputs]
        T2 = time.perf_counter()

        # save rejection trajectories
        for i, rejection_generation_output in enumerate(rejection_generation_outputs):
            record = TrajectoryRecord(
                    id=datapoint.id,
                    question=datapoint.question,
                    ground_truth=datapoint.ground_truth,
                    prompt=rejection_generation_output.text_question,
                    generated_text=rejection_generation_output.text_cot_with_answer,
                    cot_steps=rejection_generation_output.cot_steps,
                    final_answer=rejection_generation_output.final_answer,
                    evaluation_result=None,
                    confidences=rejection_confidences[i],
                    timings=Timings(
                        generation_time=T1-T0,
                        confidence_time=T2-T1,
                    )
                )
        
            self.dataset.evaluate(record)
            
            self.rejection_trajectory_repository.save(
                trajectory_record=record,
                sample=i,
            )


    def _run_generation_and_confidence_lawyer(self):
        """
        - generate lawyer samples & confidences;
        - outputs to the lawyer dir (one file per sample)
        """
        # generate lawyer samples & confidences
        datapoint = self.context.datapoint
        T0 = time.perf_counter()
        lawyer_generation_outputs: list[ParsedOutputGeneration] = self.lawyer_sampling.generate()
        T1 = time.perf_counter()
        lawyer_confidences = [self.confidence_engine.compute_confidence(output) for output in lawyer_generation_outputs]
        T2 = time.perf_counter()

        # save lawyer trajectories
        for i, lawyer_generation_output in enumerate(lawyer_generation_outputs):
            record = TrajectoryRecord(
                    id=datapoint.id,
                    question=datapoint.question,
                    ground_truth=datapoint.ground_truth,
                    prompt=lawyer_generation_output.text_question,
                    generated_text=lawyer_generation_output.text_cot_with_answer,
                    cot_steps=lawyer_generation_output.cot_steps,
                    final_answer=lawyer_generation_output.final_answer,
                    evaluation_result=None,
                    confidences=lawyer_confidences[i],
                    timings=Timings(
                        generation_time=T1-T0,
                        confidence_time=T2-T1,
                    )
            )
            
            self.dataset.evaluate(record)
            
            self.lawyer_trajectory_repository.save(
                trajectory_record=record,
                sample=i,
            )


    def _run_generation_and_confidence_stepbootstrap(self):
        """
        - generate stepbootstrap samples & confidences;
        - outputs to the stepbootstrap dir (one file per sample)
        """
        # generate stepbootstrap samples & confidences
        datapoint = self.context.datapoint
        T0 = time.perf_counter()
        stepbootstrap_generation_outputs: list[ParsedOutputGeneration] = self.stepbootstrap_sampling.generate()
        T1 = time.perf_counter()
        stepbootstrap_confidences = [self.confidence_engine.compute_confidence(output) for output in stepbootstrap_generation_outputs]
        T2 = time.perf_counter()

        # save stepbootstrap trajectories
        for i, stepbootstrap_generation_output in enumerate(stepbootstrap_generation_outputs):
            record = TrajectoryRecord(
                    id=datapoint.id,
                    question=datapoint.question,
                    ground_truth=datapoint.ground_truth,
                    prompt=stepbootstrap_generation_output.text_question,
                    generated_text=stepbootstrap_generation_output.text_cot_with_answer,
                    cot_steps=stepbootstrap_generation_output.cot_steps,
                    final_answer=stepbootstrap_generation_output.final_answer,
                    evaluation_result=None,
                    confidences=stepbootstrap_confidences[i],
                    timings=Timings(
                        generation_time=T1-T0,
                        confidence_time=T2-T1,
                    )
                )

            self.dataset.evaluate(record)

            self.stepbootstrap_trajectory_repository.save(
                trajectory_record=record,
                sample=i,
            )


    def run_generation_and_confidence(self, datapoint: Datapoint):
        self.context.datapoint = datapoint

        # 1. vanilla
        self._run_generation_and_confidence_vanilla()


        # 2. rejection
        self._run_generation_and_confidence_rejection()

        # 3 - generate lawyer samples & confidences
        self._run_generation_and_confidence_lawyer()

        # 4 - generate stepbootstrap samples & confidences
        self._run_generation_and_confidence_stepbootstrap()


        # clear the all datapoint-related fields in the context to avoid accidentally using them for the next datapoint
        self.context.clear()


    def run(self):
        if self.generation_config.from_pickle is not None:
            datapoints: list[Datapoint] = self.dataset.load_datapoints_from_pickle(self.generation_config.from_pickle)

            if self.generation_config.sample_size:
                datapoints = datapoints[:self.generation_config.sample_size]
                for datapoint in datapoints:
                    self.run_generation_and_confidence(datapoint)
        
        

        



