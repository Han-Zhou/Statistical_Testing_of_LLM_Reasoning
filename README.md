# Statistical_Testing_of_LLM_Reasoning

This directory contains all the necessary scripts for the paper "Statistical Testing of LLM Reasoning"


## 1. Repo structure

### 1.1. Voilà

```
.
├── main.py                         # CLI entry point
├── config.py                       # CLI configuration dataclasses
├── datasets/                       # Dataset loaders, prompts, and registry
│   ├── base.py
│   ├── registry.py
│   ├── bfcl.py
│   ├── bigbench_causal.py
│   ├── bigbench_movie.py
│   ├── codeqa.py
│   ├── cs1qa.py
│   ├── hotpotqa.py
│   ├── logiqa.py
│   └── math500.py
├── models/
│   ├── registry.py                # Top-level model registry
│   ├── parsing_utils.py
│   ├── adapters/                  # Backend/model-specific adapters
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── deepseek_adapter.py
│   │   ├── gpt_adapter.py
│   │   ├── llama_adapter.py
│   │   └── qwen*_adapter.py
│   └── core_models/               # API, Hugging Face, and vLLM wrappers
│       ├── api_llm.py
│       ├── llm.py
│       ├── registry.py
│       └── vllm_llm.py
├── pipeline/
│   ├── runner.py                  # Experiment orchestration
│   └── sampling/                 # Sampling strategies
│       ├── base.py
│       ├── context.py
│       ├── factory.py
│       ├── vanilla_sampling.py
│       ├── rejection_sampling.py
│       ├── lawyer_sampling.py
│       └── stepbootstrap_sampling.py
├── confidence/                     # Confidence scoring
│   ├── base.py
│   ├── confidence_engine.py
│   ├── indirect.py
│   └── verbal.py
├── evaluation/                     # Answer extraction and comparison
│   ├── comparators.py
│   └── extractors.py
├── domain/                         # Shared data contracts
│   ├── confidence.py
│   ├── data.py
│   └── evaluation.py
├── repository/
│   └── trajectory_repository.py  # Trajectory persistence
├── scripts/                        # Smoke tests and experiment launchers
├── tests/                          # Targeted pytest diagnostics
├── test_playground/                # Interactive diagnostics
├── analysis_notebooks/             # Analysis notebooks
└── trajectories/                   # Generated outputs (git-ignored)
```

(uml.png can be referenced but it is outdated)

### 1.2. Explainations

1. The entry point is at [`main.py`](main.py).
    - it will propagate to [`pipeline.Runner`](pipeline/runner.py):
        - CLI flags become `GenerationConfig`, `ConfidenceConfig`, and `SamplingConfig` ([`config.py`](config.py))
        - `Runner` builds the model adapter, dataset, `SampleContext`, and the four sampling methods via [`SamplingMethodFactory`](pipeline/sampling/factory.py)
        - `Runner.run()` loads datapoints and, for each one, runs the four methods in a fixed order (later methods reuse vanilla CoT, answer, and question cache from [`SampleContext`](pipeline/sampling/context.py))
    - in which it will perform 4 sampling methods:
        - vanilla sampling: [`vanilla_sampling.py`](pipeline/sampling/vanilla_sampling.py)
        - rejection sampling: [`rejection_sampling.py`](pipeline/sampling/rejection_sampling.py)
        - lawyer sampling: [`lawyer_sampling.py`](pipeline/sampling/lawyer_sampling.py)
        - step-bootstrap sampling: [`stepbootstrap_sampling.py`](pipeline/sampling/stepbootstrap_sampling.py)
    - in each sampling method, there are 4 scoring methods:
        - answer-token probability: [`confidence_engine.py`](confidence/confidence_engine.py)
        - answer-token entropy: [`confidence_engine.py`](confidence/confidence_engine.py)
        - indirect True/False confidence: [`indirect.py`](confidence/indirect.py)
        - verbal confidence from 0 to 100: [`verbal.py`](confidence/verbal.py)
    - output should be in  `trajectories/`


2. The models and its relevant adapters are in [`models/`](models/)
    - they are separated into categories:
        - [`core_models/`](models/core_models/) for API, Hugging Face, and vLLM wrappers
        - [`adapters/`](models/adapters/) for model-specific generation and scoring


3. The timing calculations can be found in [`pipeline/runner.py`](pipeline/runner.py) and [`confidence/confidence_engine.py`](confidence/confidence_engine.py)
    - NOTE: the api llm calculations need tweaking for more precise timing

4. Run scripts are in [`scripts/`](scripts/)
    - smoke tests (`test_*_generation.sh`) and per-model experiment launchers (`*/bm/`)
    - previous run scripts are in [`scripts/archived/`](scripts/archived/)


5. All the datasets in pickle form are in external storage paths referenced by the run scripts
    - they are loaded with `--from_pickle`; conversion scripts are in [`datasets/to_pickle/`](datasets/to_pickle/)
  
6. Evaluation (per-dataset / category) are in [`evaluation/`](evaluation/)


6. Timing analysis & other testing notebooks are in  [`analysis_notebooks/`](analysis_notebooks/)



7. Some scripts that might be useful in terms of compression, slurm stuff & post processing are in [`other_scripts/`](other_scripts/)








## Appendix

NOTES:
1. answer_prob and answer_ent extraction in T0-T1 to increase proformance
2. Due to the Gated DeltaNet structure of Qwen, need to run solo forward pass over question prompt
