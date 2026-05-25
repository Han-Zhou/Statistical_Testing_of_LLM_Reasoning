# Statistical_Testing_of_LLM_Reasoning

This directory contains all the necessary scripts for the paper "Statistical Testing of LLM Reasoning"


## 1. Repo structure
```
repo
  main.py
  config.py
  pipeline/
    runner.py
    contexts.py
    generation_engine.py
    confidence_engine.py
    evaluation_engine.py
  datasets/
    base.py
    bfcl.py
    bigbench_causal.py
    bigbench_movie.py
    codeqa.py
    cs1qa.py
    hotpotqa.py
    logiqa.py
    math500.py
    registry.py
  models/
    base.py
    registry.py
    llm_client.py
    llama_adapter.py
    qwen_adapter.py
    gpt_adapter.py
    rendering.py
  confidence/
    base.py
    engine.py


      vanilla.py
      coinflip.py
      jackknife_mask.py
      jackknife_reconstruct.py
      bootstrap.py
      fake_bootstrap.py
      hollow.py
  parsing/
    cot_parser.py
  evaluation/
    registry.py
    comparators.py
    extractors.py
  io/
    trajectory_store.py
    debug_cache_store.py
    pregenerated_store.py
  domain/
    data.py
    confidence.py
    evaluation.py
```




NOTES:
1. answer_prob and answer_ent extraction in T0-T1 to increase proformance
2. Due to the Gated DeltaNet structure of Qwen, need to run solo forward pass over question prompt
