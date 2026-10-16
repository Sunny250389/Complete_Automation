# Complete_Automation

A Python test automation framework. It is being built in phases; see the architecture plan for the full scope.

## Current state (Phase 1: shared core)

- `src/complete_automation/core/`: shared, domain-independent infrastructure. It covers configuration, logging and redaction, exceptions, execution context, the test result contract, JSON reporting, artifact storage, test data loaders, retry and pytest fixtures. See [docs/shared_core.md](docs/shared_core.md).
- `src/complete_automation/llm/`, `metrics/`, `utils/`, `config/`, `reporting/`: the existing LLM evaluation code, kept working on top of the core.
- `config/environments/`: `dev.yaml` (default) and `test.yaml`.
- `tests/unit/`: offline tests, including `tests/unit/core/` for the shared core.
- `tests/llm/`: DeepEval suites. They need a judge model and an API key to run.
- `data/llm/eval_cases.xlsx` and `eval_cases.json`: seed dataset, six cases. The two files hold the same rows.
- `RAG-Testing-Framework/`: prototype, not yet part of the package.

## Quick start

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run ruff check src tests
uv run pytest --collect-only      # collects every test without running it
uv run pytest tests/unit          # offline unit tests
uv run pytest tests/unit --cauto-report   # also writes reports/<run_id>/results.json
```

## Configuration

Settings come from `config/environments/<env>.yaml`, then `CAUTO_*` values in `.env`, then `CAUTO_*` environment variables. Later sources win. Select the environment with `CAUTO_ENVIRONMENT` or `--cauto-env`. See `.env.example` for variable names, and [docs/shared_core.md](docs/shared_core.md#configuration) for the full rules.

Never commit `.env` or real credentials.

### LLM suite variables

- `CONFIDENT_API_KEY`: optional. If set, logs into Confident AI at session start.
- `DEEPEVAL_MODEL`: judge model for DeepEval (default `gpt-4.1`).
- `CONFIDENT_ALIAS`: optional alias for Confident AI datasets and reports. Not yet used by code.
- `LOG_LEVEL`: not yet used by code. Use `CAUTO_LOGGING__LEVEL` for framework logging.
