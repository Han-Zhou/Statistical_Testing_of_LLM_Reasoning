import argparse
import logging

from dotenv import load_dotenv

from config import ConfidenceConfig, GenerationConfig, SamplingConfig
from pipeline import Runner


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse():
    args = argparse.ArgumentParser(description="Statistical Testing of LLM Reasoning")
    args.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        required=True,
        help="Backend for model inference. vLLM is faster but incompatible with --confidence.",
    )
    args.add_argument(
        "--confidence",
        type=str,
        default="none",
        choices=["none", "vanilla", "step_bootstrap"],
        help="Whether or not to evaluate confidence on the COTs."
    )
    args.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["bfcl", "bigbench_movie", "bigbench_causal", "logiqa", "codeqa", "cs1qa", "hotpotqa", "math500"],
        help=(
            "Dataset to process. One of: bfcl, bigbench_movie, "
            "bigbench_causal, logiqa, codeqa, cs1qa, hotpotqa, math500."
        )
    )
    args.add_argument(
        "--discord",
        action="store_true",
        default=False,
        help="Send tqdm progress to Discord via TQDM_DISCORD_TOKEN and TQDM_DISCORD_CHANNEL_ID env vars."
    )
    args.add_argument(
        "--from_pickle",
        type=str,
        default=None,
        help="Path to a pickle file containing raw dataset entries (NOT pre-generated). --dataset is still required for prompt/eval routing."
    )
    args.add_argument(
        "--from_pregenerated",
        type=str,
        default=None,
        help="Path to a pickle file containing pre-generated entries. --dataset is still required for prompt/eval routing."
    )
    args.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="Maximum number of new tokens to generate for each prompt."
    )
    args.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llama", "qwen"],
        help="The model to use for the statistical testing"
    )
    args.add_argument(
        "--nb_cot_samples",
        type=int,
        default=1,
        help="Number of CoT samples per datapoint. Requires temperature > 0 for diversity.",
    )
    args.add_argument(
        "--nb_dropout_samples",
        type=int,
        default=None,
        help="Number of dropout samples for confidence scoring. Defaults to 3. Must not be set with --vanilla_only."
    )
    args.add_argument(
        "--sample_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Slice [start, end) of the dataset to process. Mutually exclusive with --sample_size."
    )
    args.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process from the dataset. Defaults to all."
    )
    args.add_argument(
        "--seed_stepbootstrap",
        type=int,
        default=0,
        help="Seed for random number generation for StepBootstrap sampling. Defaults to 0."
    )
    args.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional prefix for the output directory name."
    )
    args.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. 0.0 = greedy decoding.",
    )
    args.add_argument(
        "--prompt_type",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "Prompt type. "
            "1 (default): assistant prefill with 'Step 1:'. "
            "2: 'Let's think step-by-step.' in user prompt, no assistant prefill."
        ),
    )

    return args







def main():
    args = parse().parse_args()
    confidence_config, generation_config, sampling_config = ConfidenceConfig.from_args(args), GenerationConfig.from_args(args), SamplingConfig.from_args(args)
    runner = Runner(
        generation_config=generation_config, 
        confidence_config=confidence_config,
        sampling_config=sampling_config
    )
    runner.run()
    

if __name__ == "__main__":
    main()











