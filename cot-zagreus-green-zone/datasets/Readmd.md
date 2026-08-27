`datasets/` contains the dataset adapters used to load datapoints, build prompt messages, and evaluate trajectories for each supported benchmark.

Files:
* `base.py` --> defines the abstract `Dataset` base class that every benchmark adapter must implement:
- `load_datapoints` for loading raw datapoints (e.g. fetched from source)
- `load_datapoints_from_pickle` for loading datapoints from a pre-cached pickle
- `build_messages` for assembling the chat messages from a datapoint and a `PromptRequest`
- `evaluate` for scoring a `TrajectoryRecord` against ground truth
- `resolve_pregenerated` for hydrating pre-generated trajectories from a pickle
* `registry.py` --> contains the `DATASETS` registry that maps a dataset name to its concrete `Dataset` subclass (used by `main.py` to resolve the adapter from config)
* `bigbench_movie.py` --> `BigBenchMovieDataset` adapter for the BBH Movie Recommendation task (MCQ, exact-match evaluation on the boxed letter)
* `bigbench_causal.py` --> `BigBenchCausalDataset` adapter for the BBH Causal Judgment task
* `bfcl.py` --> `BfclDataset` adapter for the Berkeley Function Calling Leaderboard
* `codeqa.py`, `cs1qa.py`, `hotpotqa.py`, `logiqa.py`, `math500.py` --> placeholders for upcoming benchmark adapters
* `to_pickle/` --> scripts that subsample (when a dataset exceeds ~1000 datapoints) and cache datasets into pickle files consumed by `load_datapoints_from_pickle`
