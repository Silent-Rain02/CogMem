# Reproducibility guide

This guide documents the evaluation runners included in this repository. A successful smoke test only verifies the pipeline; it does not reproduce the reported paper numbers.

## Environment

Use Python 3.12 and install the locked project environment:

```bash
uv sync --extra dev
cp .env.example .env
```

The camera-ready configuration uses BGE-M3 embeddings (1024 dimensions), temperature `0.0`, up to 15 anchors, traversal depth 2, up to 15 neighbors per hop, and at most 5 controller calls. The checked-in defaults are runtime-oriented; set the paper values explicitly in your experiment configuration and record every model endpoint and version.

```python
from cogmem import CogMemConfig

config = CogMemConfig()
config.model.temperature = 0.0
config.agent.find_anchors_top_k = 15
config.agent.traverse_neighbors_max_depth = 2
config.agent.traverse_neighbors_max_results = 15
config.agent.max_iterations = 5
```

The reported experiments used GPT-4o-mini and Qwen2.5-14B-Instruct. Qwen2.5-14B-Instruct was served locally; GPT-4o-mini was accessed through its API. Exact outputs may vary with serving stack, model revision, provider behavior, and benchmark release.

## LoCoMo

Obtain `locomo10.json` from the official LoCoMo distribution and retain its license and attribution. Run a one-question smoke test before starting a complete experiment:

```bash
uv run python scripts/eval_locomo_complete.py \
  --data_dir /path/to/locomo10.json \
  --workspace .cogmem/locomo_smoke \
  --max_samples 1 \
  --output artifacts/locomo_smoke.json
```

Run the full pipeline with a clean workspace:

```bash
uv run python scripts/eval_locomo_complete.py \
  --data_dir /path/to/locomo10.json \
  --workspace .cogmem/locomo \
  --output artifacts/locomo_results.json
```

The runner builds the graph, runs offline consolidation, writes per-question traces, and reports EM, F1, BLEU-1, evidence hit, and optional LLM-judge results. Use `--skip_index` only with the same dataset/model/workspace, and `--skip_consolidation` only for the corresponding ablation.

## LongMemEval

Prepare the benchmark JSON in the format expected by `datasets/longmemeval_loader.py`, then run:

```bash
uv run python scripts/eval_longmemeval_complete.py \
  --data_dir /path/to/longmemeval_s_cleaned.json \
  --workspace .cogmem/longmemeval \
  --output artifacts/longmemeval_results.json
```

The provided runner writes lexical metrics alongside predictions. To reproduce the paper's LongMemEval headline metric, run the benchmark's official LLM-as-judge protocol on the generated predictions and record judge model, prompt, and seed.

## Report what you ran

For each experiment, save the command line, Git commit, dataset checksum, model names and revisions, API/serving configuration, random seed, and full output JSON. Do not publish API keys, raw private conversations, or persisted memory workspaces.
