# Complete_Automation

A Python test automation framework. It is being built in phases; see the architecture plan for the full scope.

## Current state (Phase 0)

- `src/complete_automation/`: the LLM evaluation package, moved from `LLM_Automation/` unchanged apart from import paths. It is restructured in Phase 1.
- `tests/llm/`: DeepEval suites. They need a judge model and API key to run.
- `tests/unit/`: offline checks on the dataset and xlsx reader.
- `data/llm/eval_cases.xlsx` and `eval_cases.json`: seed dataset, six cases. The two files hold the same rows.
- `RAG-Testing-Framework/`: prototype, not yet part of the package. It is migrated in Phase 4.

## Quick start

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run ruff check src tests
uv run pytest --collect-only -q   # collects every test without running it
uv run pytest tests/unit          # offline unit tests
```

## Environment variables

Set these in a local `.env` file. Never commit `.env`.

- `CONFIDENT_API_KEY`: optional. If set, logs into Confident AI at session start.
- `DEEPEVAL_MODEL`: judge model for DeepEval (default `gpt-4.1`).
- `CONFIDENT_ALIAS`: optional alias for Confident AI datasets and reports. Not yet used by code.
- `LOG_LEVEL`: logging level (default `INFO`). Not yet used by code.
