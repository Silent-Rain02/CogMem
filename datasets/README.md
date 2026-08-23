# Dataset notes

CogMem provides **adapters** for LoCoMo and LongMemEval; the benchmarks remain their respective authors' datasets. Use the original sources and licenses when downloading, sharing, or modifying benchmark data.

- LoCoMo: [paper](https://aclanthology.org/2024.acl-long.747/) and [official repository](https://github.com/snap-research/locomo)
- LongMemEval: [paper](https://openreview.net/forum?id=pZiyCaVuti)

No benchmark data is tracked or redistributed by this repository. Download each benchmark from its official source, respect its license and attribution requirements, and pass the local path to the corresponding evaluator.

The evaluators accept an explicit path, so a dataset does not need to live in this directory:

```bash
uv run python scripts/eval_locomo_complete.py --data_dir /path/to/locomo10.json
uv run python scripts/eval_longmemeval_complete.py --data_dir /path/to/longmemeval_s_cleaned.json
```
