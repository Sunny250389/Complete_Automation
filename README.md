# Complete_Automation

A Python test automation framework. It is being built in phases; see the architecture plan for the full scope.

## Current state (shared core, API automation, database testing and UI automation)

- `src/complete_automation/core/`: shared, domain-independent infrastructure. It covers configuration, logging and redaction, exceptions, execution context, the test result contract, JSON reporting, artifact storage, test data loaders, retry and pytest fixtures. See [docs/shared_core.md](docs/shared_core.md).
- `src/complete_automation/api/`: API automation (optional `api` extra). It provides HTTPX sync and async clients, authentication (API key, bearer, basic, OAuth2 client credentials), assertions, JSON Schema and OpenAPI contract validation, safe retries, mocking, and fixtures. See [docs/api_automation.md](docs/api_automation.md).
- `src/complete_automation/database/`: database testing (optional `database` and `postgresql` extras). It provides SQLite and PostgreSQL adapters, safe parameterized SQL execution, explicit transactions, read-only and destructive-operation safeguards, CRUD, integrity and schema validation, reconciliation, and fixtures. See [docs/database_testing.md](docs/database_testing.md).
- `src/complete_automation/ui/`: UI automation (optional `ui` extra). It provides Playwright browser lifecycle (Chromium, Firefox, WebKit), page objects, UI assertions, per-test failure diagnostics (screenshots, console log, optional trace and video) and fixtures. See [docs/ui_automation.md](docs/ui_automation.md).
- `tests/integration/`, `tests/examples/`: API tests against a local test service on 127.0.0.1, database tests against temporary SQLite databases, and UI tests in a real browser against a local practice site.
- `tests/live/`: opt-in tests against real external APIs; they skip unless configured.
- `src/complete_automation/llm/`, `metrics/`, `utils/`, `config/`, `reporting/`: the existing LLM evaluation code, kept working on top of the core.
- `config/environments/`: `dev.yaml` (default), `test.yaml`, `postgres_test.yaml` (opt-in PostgreSQL tests), and `ui_local.yaml` (example for UI tests against a local application).
- `tests/unit/`: offline tests, including `tests/unit/core/` for the shared core.
- `tests/llm/`: DeepEval suites. They need a judge model and an API key to run.
- `data/llm/eval_cases.xlsx` and `eval_cases.json`: seed dataset, six cases. The two files hold the same rows.
- `RAG-Testing-Framework/`: prototype, not yet part of the package.

## Quick start

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-extras     # or --extra api, --extra ui; plain "uv sync --locked" installs the core only
uv run playwright install chromium   # browser for the UI tests (or firefox, webkit)
uv run ruff check src tests
uv run pytest --collect-only      # collects every test without running it
uv run pytest tests/unit          # offline unit tests
uv run pytest -m api              # every API test (live ones skip unless configured)
uv run pytest tests/integration tests/examples   # API tests against the local test service, database tests on SQLite
uv run pytest -m ui               # every UI test (needs the ui extra and an installed browser)
uv run pytest -m database         # every database test (PostgreSQL ones skip unless configured)
uv run pytest -m postgresql --cauto-env postgres_test   # PostgreSQL tests; see docs/database_testing.md
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
