from types import MappingProxyType

from datasets.bigbench_movie import BigBenchMovieDataset
from datasets.bigbench_causal import BigBenchCausalDataset
from datasets.logiqa import LogiQADataset
from datasets.hotpotqa import HotPotQADataset
from datasets.math500 import Math500Dataset
from datasets.codeqa import CodeQADataset
from datasets.cs1qa import CS1QADataset
from datasets.bfcl import BfclDataset

DATASETS = MappingProxyType({
    "bigbench_movie": BigBenchMovieDataset,
    "bigbench_causal": BigBenchCausalDataset,
    "logiqa": LogiQADataset,
    "hotpotqa": HotPotQADataset,
    "math500": Math500Dataset,
    "codeqa": CodeQADataset,
    "cs1qa": CS1QADataset,
    "bfcl": BfclDataset,
})
