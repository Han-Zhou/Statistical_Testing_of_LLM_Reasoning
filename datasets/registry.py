from types import MappingProxyType

from datasets.bigbench_movie import BigBenchMovieDataset

DATASETS = MappingProxyType({
    "bigbench_movie": BigBenchMovieDataset
})

