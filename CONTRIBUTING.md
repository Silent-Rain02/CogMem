# Contributing

Thank you for improving CogMem. Until the public release, please coordinate changes privately with the maintainers.

## Development workflow

1. Create an isolated environment with `uv sync --extra dev`.
2. Keep changes focused and add an offline test for behavior that does not require a model endpoint.
3. Run `uv run pytest` and `uv run ruff check cogmem datasets scripts examples tests` before opening a review.
4. Do not commit `.env`, model credentials, raw dialogue, persisted workspaces, or benchmark outputs containing private content.

Changes to data adapters or evaluation logic should state the benchmark version, expected input format, and compatibility impact. Changes that modify reported results require a reproducible command and saved evaluation artifact.
