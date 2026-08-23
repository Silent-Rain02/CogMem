# CogMem: From Retrieval to Reconstruction

CogMem is a provenance-aware cognitive memory system for long-term dialogue. It turns dialogue into an evolvable **PEC²F** graph—**P**erson, **E**vent, **C**oncept, **C**laim, and **F**act—and uses a bounded agentic search loop to reconstruct answers from grounded evidence.

> **Pre-release notice.** This repository is being prepared for public release. The paper link and citation are intentionally omitted until the publication is formally available. Please do not treat this branch as the archival release.

[中文说明](README_zh.md)

<p align="center">
  <img src="assets/cogmem_overview.png" alt="CogMem architecture: memory generation, consolidation, and cognitive reasoning" width="100%">
</p>

## Why CogMem?

Flat retrieval can conflate a speaker's attributed opinion with an event or a durable fact. CogMem keeps those forms of memory distinct and connects them with typed, time-aware edges.

| Stage | What happens |
| --- | --- |
| Memorize | Dialogue turns are extracted into typed graph records with source and timestamp metadata. |
| Consolidate | Episodic events are synthesized into durable facts while retaining links to their supporting evidence. |
| Reconstruct | A query-aware controller composes anchoring, traversal, intersection, and evidence-grounding operators. |

The design supports multi-hop reasoning, temporal queries, knowledge updates, and source-attributed claims without training a retrieval policy.

## Installation

CogMem requires Python 3.12 or later. We recommend [`uv`](https://docs.astral.sh/uv/) for reproducible environments, although `pip` works as well.

```bash
git clone https://github.com/Silent-Rain02/CogMem.git
cd CogMem
uv sync
# or: python -m pip install -e .
```

Copy the configuration template and choose one supported model provider:

```bash
cp .env.example .env
```

CogMem supports OpenAI-compatible APIs, SiliconFlow, ModelScope, and Ollama. Configure the provider and model identifiers in your application; see [configuration](#configuration). Keep `.env` private—API credentials must never be committed.

## Quick start

The following example requires a configured chat model and embedding endpoint. It creates a fresh local workspace and persists the resulting memory graph there.

```python
from cogmem import CogMem, CogMemConfig
from cogmem.storage import Doc

config = CogMemConfig()
config.model.provider = "ollama"
config.model.embedding_provider = "ollama"
config.model.ollama_chat_model = "qwen2.5:14b-instruct"
config.model.ollama_embedding_model = "bge-m3"

memory = CogMem(config=config, workspace_dir="./workspace")
memory.add_documents([
    Doc(id="turn-1", text="Alice and Bob visited the park on Saturday."),
    Doc(id="turn-2", text="Bob said that he enjoys birdwatching."),
])
memory.save()

print(memory.answer("What does Bob enjoy?"))
```

For configuration of an OpenAI-compatible endpoint, see [`.env.example`](.env.example). A runnable copy of this snippet is available at [`examples/quickstart.py`](examples/quickstart.py).

## Configuration

`CogMemConfig` keeps model, storage, extraction, and agent settings separate. Start from an independent config object rather than mutating `DEFAULT_CONFIG`:

```python
from cogmem import CogMemConfig

config = CogMemConfig()
config.model.provider = "siliconflow"
config.model.embedding_provider = "siliconflow"
config.model.siliconflow_chat_model = "Qwen/Qwen2.5-14B-Instruct"
config.model.siliconflow_embedding_model = "BAAI/bge-m3"
config.runtime.max_concurrency = 2
```

The implementation detects the embedding dimension before creating its FAISS indexes. Rebuild a workspace when switching to an embedding model with a different output dimension.

## Reproducing the experiments

CogMem includes runners for LoCoMo and LongMemEval. The camera-ready setting uses BGE-M3 embeddings (1024 dimensions), temperature `0.0`, 15 anchors, traversal depth 2, up to 15 neighbors per hop, and at most 5 controller calls:

```python
from cogmem import CogMemConfig

config = CogMemConfig()
config.model.temperature = 0.0
config.agent.find_anchors_top_k = 15
config.agent.traverse_neighbors_max_depth = 2
config.agent.traverse_neighbors_max_results = 15
config.agent.max_iterations = 5
```

Download benchmark data from the official sources before running an evaluation: [LoCoMo paper](https://aclanthology.org/2024.acl-long.747/), [LoCoMo repository](https://github.com/snap-research/locomo), and [LongMemEval paper](https://openreview.net/forum?id=pZiyCaVuti). This repository does not distribute benchmark data.

```bash
# LoCoMo: build the memory graph, consolidate it, and evaluate.
uv run python scripts/eval_locomo_complete.py \
  --data_dir /path/to/locomo10.json \
  --workspace .cogmem/locomo \
  --output artifacts/locomo_results.json

# LongMemEval: use the benchmark's prepared JSON file.
uv run python scripts/eval_longmemeval_complete.py \
  --data_dir /path/to/longmemeval_s_cleaned.json \
  --workspace .cogmem/longmemeval \
  --output artifacts/longmemeval_results.json
```

Start with a one-question LoCoMo smoke test before a full run by adding `--max_samples 1`. Exact outputs can vary with the model revision, provider/serving stack, dataset release, prompt, and judge configuration. Record the Git commit, dataset checksum, model identifiers, command, and full output JSON for every reported run. LongMemEval's paper metric requires its official LLM-as-judge protocol on the generated predictions.

## Reported results

The camera-ready experimental record reports the following CogMem results. LoCoMo values are F1 / BLEU-1; LongMemEval values are LLM-as-judge accuracy. Results depend on the listed backbone, dataset version, prompts, and evaluation protocol.

| Benchmark | GPT-4o-mini | Qwen2.5-14B-Instruct |
| --- | ---: | ---: |
| LoCoMo — Single Hop | 59.20 / 56.40 | 60.40 / 55.80 |
| LoCoMo — Multi Hop | 50.70 / 46.10 | 48.42 / 44.65 |
| LoCoMo — Temporal | 63.80 / 57.90 | 56.20 / 49.10 |
| LoCoMo — Open Domain | 56.10 / 51.20 | 55.80 / 50.10 |
| LongMemEval — Accuracy | 68.40 | 66.50 |

## Repository layout

```text
cogmem/       Core library: graph schema, ingestion, consolidation, and agents
datasets/     Dataset adapters and local-development dataset notes
examples/     Small runnable examples
scripts/      Benchmark evaluation runners
tests/        Fast, offline unit tests
```

## Safety and privacy

CogMem stores potentially sensitive dialogue content, relationships, preferences, and attributed opinions. Use per-user workspaces, control access to persisted graph files, avoid logging raw conversations, and offer retention/deletion controls in any deployment. Report security issues and leaked credentials privately to the maintainers rather than opening a public issue.

## Contributing

Until public release, coordinate changes privately with the maintainers. Before publishing this repository, confirm the paper's publication status; add its final URL and BibTeX; keep benchmark data, credentials, private conversations, and uncertain media out of Git history; run `uv sync --extra dev`, `uv run pytest`, and the documented smoke tests from a clean checkout; then verify every README result against the archived experiment outputs.

## License

CogMem is released under the [MIT License](LICENSE).
