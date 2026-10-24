# Integration, test execution and CI (Phase 7)

Phase 7 makes the six modules work as one framework without merging them: one pytest workflow, workflows that combine modules against a local application, one correlated report per run, and a CI pipeline. It adds no new testing module and changes no module's behaviour. Docker, Docker Compose, Kubernetes and other container tooling are out of scope and are not used anywhere.

## Phases and responsibilities

| Phase | Package | Responsibility | Guide |
| --- | --- | --- | --- |
| 1 | `complete_automation.core` | Configuration, logging and redaction, exceptions and error categories, run and test context, the `TestResult` contract, JSON reporting, artifacts, test data, retry, pytest fixtures | [shared_core.md](shared_core.md) |
| 2 | `complete_automation.api` | HTTP clients, authentication, assertions, JSON Schema and OpenAPI contracts, mocking | [api_automation.md](api_automation.md) |
| 3 | `complete_automation.database` | SQLite and PostgreSQL adapters, safe SQL, transactions, data and schema validation, reconciliation | [database_testing.md](database_testing.md) |
| 4 | `complete_automation.ui` | Playwright browser lifecycle, page objects, UI assertions, failure diagnostics | [ui_automation.md](ui_automation.md) |
| 5 | `complete_automation.ml` | Evaluation of classification, regression and clustering models, quality gates, baselines | [ml_evaluation.md](ml_evaluation.md) |
| 6 | `complete_automation.llm_evaluation` | Real Ollama model calls, deterministic checks, optional judge model, quality gates, baselines | [llm_evaluation.md](llm_evaluation.md) |
| 7 | tests, CI, `core.summary` | Cross-module workflows, unified selection and reporting, CI pipeline | this page |

`src/complete_automation/llm`, `metrics`, `utils`, `config` and `reporting`, with `tests/llm`, are the older DeepEval prototype. They are kept working and are not part of the six modules.

## Integration boundaries

```
                     complete_automation.core
      (config, context, results, reporting, artifacts, exceptions, pytest plugin)
         ^            ^            ^            ^             ^
         |            |            |            |             |
        api       database        ui           ml      llm_evaluation
```

- Every module imports the core and nothing else from the package. The core imports no module. Modules never import each other.
- These rules are enforced: `tests/unit/<module>/test_architecture.py` checks each file's imports, and `tests/unit/test_module_independence.py` imports every module and its pytest plugin in a fresh interpreter and fails if another module is loaded, which also rules out import cycles.
- Code that combines modules lives in tests only: `tests/integration/cross_module/` and the application it drives, `tests/support/users_app.py`. There is no integration package in `src`, because the shared core already provides what an integration layer would need (below).

### Shared interfaces the integration uses

| Need | Shared-core interface |
| --- | --- |
| Configuration | `FrameworkSettings.section(name, Model)` per module, loaded once by the `framework_settings` fixture |
| Run correlation | `RunContext.run_id` (one per pytest session; `CAUTO_EXECUTION__RUN_ID` fixes it), `TestContext.correlation_id` per test |
| Results | One `TestResult` per test, with metrics, metadata and artifacts collected through the `result_recorder` fixture |
| Reporting | `JsonReporter` writes `reports/<run_id>/results.json`; `summarize` counts per module; `core.summary` renders it |
| Artifacts | `LocalArtifactStore` under `artifacts/<run_id>/<module>/...` |
| Error categories | `ErrorCategory` (`assertion`, `threshold`, `infrastructure`, `timeout`, `configuration`, `test_data`, `unknown`) |
| Test data | `DataFactory(seed)` for deterministic, unique values |

### Module interfaces (as audited)

| Module | Public entry points | Pytest fixtures (plugin) |
| --- | --- | --- |
| core | `load_settings`, `RunContext`, `TestContext`, `TestResult`, `JsonReporter`, `LocalArtifactStore`, `DataFactory`, `RetryPolicy` | `framework_settings`, `run_context`, `artifact_store`, `test_context`, `result_recorder` |
| api | `ApiClient`, `AsyncApiClient`, `ApiSettings`, auth classes, `api.assertions`, `OpenApiContract` | `api_settings`, `api_client_factory`, `api_client`, `openapi_contract`, `mock_api` |
| database | `EngineManager`, `DatabaseSession`, `DatabaseSettings`, `database.validations`, `reconcile` | `database_manager`, `db_session_factory`, `db_session`, `sqlite_database`, `example_session`, `postgres_session` |
| ui | `BrowserSession`, `BasePage`, `ui.assertions`, `UiSettings` | `ui_settings`, `ui_session`, `ui_context`, `ui_page`, `ui_base_url` |
| ml | `Evaluator`, `EvaluationConfig`, `ThresholdRule`, adapters, `Baseline` | `ml_settings`, `ml_dataset_loader`, `ml_load_baseline`, `ml_evaluate` |
| llm_evaluation | `OllamaClient`, `EvaluationRunner`, `JudgeEvaluator`, `EvaluationConfig`, `Baseline` | `llm_settings`, `llm_client`, `llm_live_model`, `llm_live_judge`, `llm_evaluate`, `llm_load_dataset` |

**Plugin registration.** `tests/conftest.py` registers the core and UI plugins once for the whole run. Module test folders star-import their module's plugin in their `conftest.py`. The API and database plugins both define a `pytest_runtest_makereport` hook; star-importing both into one namespace would keep only the second and silently switch off the API module's capture of exchanges on failure. `tests/integration/cross_module/conftest.py` therefore registers both as plugins, so each keeps its hook. Do the same in any folder that needs several module plugins.

## Running tests

Test folders give the tier, markers give the module. Commands below run from the repository root after `uv sync --locked --all-extras`; the browser tests also need `uv run playwright install chromium`.

| Tier | Folder | Needs |
| --- | --- | --- |
| Unit | `tests/unit/` | Nothing external: no network, browser, database server or model |
| Integration | `tests/integration/` | Local only: services on 127.0.0.1, temporary SQLite files, headless browser |
| Examples | `tests/examples/` | Same as integration; runnable documentation |
| Live | `tests/live/`, `tests/llm/` | A real external service; skipped unless configured |

| Marker | Selects |
| --- | --- |
| `api`, `database`, `ui`, `ml`, `llm_evaluation` | A module's tests in every tier |
| `cross_module` | Tests that combine modules (the workflows, unified reporting, module independence) |
| `integration` | Tests that use a local service |
| `live` | Tests against a real external service (live APIs, the DeepEval judge) |
| `llm_live` | Tests that call a real Ollama model |
| `postgresql` | Tests that need a configured PostgreSQL server |

| Goal | Command |
| --- | --- |
| Offline unit suite (all modules) | `uv run pytest tests/unit` |
| Shared core only | `uv run pytest tests/unit/core` |
| One module, every tier | `uv run pytest -m api` (or `database`, `ui`, `ml`, `llm_evaluation`) |
| One module, unit tests only | `uv run pytest tests/unit/api` |
| Cross-module tests | `uv run pytest -m cross_module` |
| Cross-module tests without a browser | `uv run pytest -m "cross_module and not ui"` |
| Browser tests | `uv run pytest -m ui tests/integration tests/examples` |
| ML evaluation | `uv run pytest -m ml` |
| LLM evaluation, no model needed | `uv run pytest -m "llm_evaluation and not llm_live"` |
| Real Ollama model | see [Real Ollama tests](#real-ollama-tests) |
| Everything local (what CI runs) | `uv run pytest tests/unit` then `uv run pytest tests/integration tests/examples` |
| Full regression | `uv run pytest` (service-dependent tests skip with their reason) |

Add `--cauto-report` (or `--cauto-env test`, which turns it on) to write `reports/<run_id>/results.json`, and `--junitxml=reports/junit.xml` for a JUnit file.

### Real Ollama tests

Real-model tests are never part of the default runs. They need Ollama running and a model installed (`ollama pull qwen3:4b`):

```bash
export CAUTO_MODULES__LLM_EVALUATION__MODEL=qwen3:4b         # the model under test
export CAUTO_MODULES__LLM_EVALUATION__JUDGE_MODEL=qwen3:4b   # optional judge; may be another model
export CAUTO_MODULES__LLM_EVALUATION__REQUIRE_MODEL=true     # an unavailable model fails instead of skipping
uv run pytest -m llm_live --cauto-env ollama_local --cauto-report
```

Without `REQUIRE_MODEL=true` an unreachable server or missing model skips the tests with the reason; with it, they fail. The response evaluated is always the model's own; nothing substitutes a recorded or mocked response.

## Cross-module workflows

Both workflows drive `tests/support/users_app.py`, a small application written for these tests (no framework, standard library only). It stores users in one SQLite file, exposes a JSON API under `/api/users` and renders an HTML list at `/users` with Promote and Delete buttons. It binds to 127.0.0.1 and reaches nothing outside the machine. Its docstring documents the schema and endpoints.

**Test data and isolation.** Every test starts its own application with its own database file in the test's temporary directory, seeded with two unrelated users (`SEED_USERS`). Records the test creates come from `DataFactory(seed)`, so they are deterministic and unique. Each workflow checks that the seed users are untouched. Cleanup is the removal of the application and its file, which the fixtures do even when the test fails; the database engine is disposed first so the file can be removed on Windows.

**Verification is read-only.** The database module opens the application's file with `access: read_only` (SQLite `mode=ro`): the application is the only writer, and a faulty test cannot change the data it verifies. Each verification runs in its own `session.transaction()`, so no read lock is held while the application writes.

### Workflow A: API and database consistency

`tests/integration/cross_module/test_api_database_consistency.py` (markers `api`, `database`, `integration`, `cross_module`):

- A user created through the API has the response structure, `Location` header and values that the database row holds, field by field.
- The API's user list reconciles with the table (`database.reconcile`, keyed on `id`).
- Rejected requests (422 validation, 409 duplicate email) change nothing in the database.
- Updates and deletes through the API are reflected in the database, and a deleted user is 404 everywhere.
- The verification connection itself cannot write.

### Workflow B: API, database and browser

`tests/integration/cross_module/test_api_database_ui_workflow.py` (also marked `ui`):

1. Create a user through the API and verify the row in the database.
2. Open `/users` in the browser (`UsersPage`, a `BasePage`) and find the record with the persisted values.
3. Promote it with the page's button; verify the new role through the API and the database.
4. Delete it with the page's button; verify it is gone from the page, the API (404) and the database, and that the seed users remain.

A second test deletes through the API and checks that the page no longer lists the user.

**What a failure produces.** A failing workflow test is one `TestResult` with the API exchanges (`api/*.json`), the database statement history (`database/queries.json`) and, for the browser workflow, a screenshot and `diagnostics.json`, all under `artifacts/<run_id>/`.

### Unified reporting check

`tests/integration/cross_module/test_unified_reporting.py` runs a generated project that registers the core, API, database, ML and LLM evaluation plugins together and runs passing and failing tests of each. It checks the one report it writes: every result and artifact carries the same run ID; each module has its own counts; an API or database check failure is `failed`/`assertion`, an ML gate miss is `failed`/`threshold` with `ml.accuracy` and its threshold kept, an unreachable model is `error`/`infrastructure`, a judge that returns no verdict is `error` (never a pass) with the target and judge models named separately, an opt-in real-model test is `skipped` with its reason; and the session exits with pytest's "tests failed" status. The models and HTTP service there are scripted transports: it proves the integration, not a model's quality.

## Reports and outcomes

Every test of a run produces one `TestResult` in `reports/<run_id>/results.json`, with `module` from its `cauto_module` marker and its other markers as `tags`. How an outcome is reported:

| Situation | `status` | `failure.category` |
| --- | --- | --- |
| Every check passed | `passed` | |
| A check failed (API, database, UI assertion) | `failed` | `assertion` |
| An ML or LLM quality gate was missed | `failed` | `threshold` |
| A required service failed or was unreachable | `error` | `infrastructure` or `timeout` |
| Bad configuration or test data | `error` | `configuration` or `test_data` |
| An evaluator (for example an LLM judge) reached no verdict | `error` | `unknown` (`EvaluatorError`) |
| The test ran but reached no verdict | `incomplete` | |
| Not run: opt-in test not enabled, or prerequisite not configured | `skipped` | reason in `metadata.skip_reason` |

ML and LLM results keep their own metrics (`ml.accuracy`, `llm.case_pass_rate`, `llm.judge.<criterion>.pass_rate`, ...) with thresholds and verdicts, and their evaluation summary under `metadata.ml_evaluation` or `metadata.llm_evaluation`. Nothing is combined into a single score.

**Summary.** `python -m complete_automation.core.summary` renders one or more reports as Markdown: totals, one row per module, then the tests that did not pass grouped as quality-gate failures, assertion failures, service or infrastructure errors, other errors and incomplete (with category, duration, message, failed metrics and artifacts), and skips grouped by reason. It reads the report and recounts with the core's `summarize`; it changes no verdict.

```bash
uv run python -m complete_automation.core.summary "reports/<run_id>/results*.json"
```

**Skipped versus blocked.** A test that cannot run because a prerequisite is missing is reported as `skipped` with the reason (for example "PostgreSQL is not configured"); the summary groups skips by reason so they stay visible. Where a missing prerequisite must fail the run, the module has a switch: `modules.llm_evaluation.require_model: true` turns an unavailable model into a failure.

## Configuration

All modules read the shared configuration: `config/environments/<env>.yaml`, then `CAUTO_*` values in `.env`, then `CAUTO_*` environment variables; later sources win (full rules in [shared_core.md](shared_core.md#configuration)). There is no second configuration system, apart from the legacy DeepEval suite's own `src/complete_automation/config/settings.py`.

| Environment | Use |
| --- | --- |
| `dev` (default) | Local development; no report unless `--cauto-report` |
| `test` | CI: JSON logs, JSON report on |
| `ollama_local` | Real-model LLM tests (`live: true`); the model comes from the environment |
| `postgres_test` | PostgreSQL tests on a disposable database; credentials from the environment |
| `ui_local` | Example: UI tests against your own application |
| `reqres` | Live API tests against reqres.in; API key from the environment |

| Tests | Required configuration |
| --- | --- |
| Unit, integration, examples, cross-module | None |
| Browser tests | `playwright install chromium` |
| Real Ollama | Ollama running; `CAUTO_MODULES__LLM_EVALUATION__MODEL` (and optionally `__JUDGE_MODEL`) |
| PostgreSQL | `--cauto-env postgres_test` and `CAUTO_MODULES__DATABASE__USERNAME` / `__PASSWORD` |
| Live API | `--cauto-env reqres` and `CAUTO_MODULES__API__AUTH__VALUE` |
| DeepEval suite (`tests/llm`) | `uv sync --extra deepeval` and `OPENAI_API_KEY` (DeepEval's default judge) |

Secrets go in environment variables or a local `.env` (git ignores it), never in YAML. Each module validates its section when a test first needs it, so a module's tests never require another module's settings.

## CI pipeline

GitHub Actions, `.github/workflows/ci.yml`. It runs on pushes to `main`, on pull requests and on demand, on `ubuntu-24.04` with Python 3.12 (the lowest version the project allows), with read-only repository permissions and no secrets.

| Job | Runs | Needs |
| --- | --- | --- |
| Static validation | `uv sync --locked --all-extras`, `ruff check`, `compileall`, `pytest --collect-only` | |
| Offline unit tests | `pytest tests/unit --cauto-env test` | Static validation |
| Local integration tests | `playwright install --with-deps chromium`, then `pytest tests/integration tests/examples --cauto-env test` (module integration, cross-module workflows, examples) | Static validation |

The unit and integration jobs run in parallel. Each writes a JUnit file and the JSON report, adds the Markdown summary to the job page, and uploads `reports/` (14 days); the integration job also uploads `artifacts/` (7 days): API exchanges and database histories of failed tests, screenshots, ML and LLM evaluation files. A failing test fails its job and the pipeline; the summary and upload steps run anyway.

**Not run in CI**, because the runners have no such service: real Ollama models, PostgreSQL, live external APIs and the DeepEval judge. The CI job installs every extra (`--all-extras`, which includes `deepeval`), so those tests are collected and reported as skipped with their reason; run them locally with the commands above. Dependencies come from `uv.lock` (`--locked`); the uv cache is keyed on it.

### Where output accumulates

Every run that writes a report gets its own `reports/<run_id>/` and `artifacts/<run_id>/` folder (both ignored by git). Nothing deletes them: the folders of earlier runs are what lets you compare runs. Remove old ones when they are no longer needed, for example `rm -rf reports/* artifacts/*`. Tests that start their own pytest session (plugin self-tests) write into pytest's temporary directory, which pytest prunes itself (it keeps the last three).

## Diagnosing a failure

1. In CI, open the job summary: the failing test, its category, message and artifact names.
2. Download `integration-report` or `unit-report` (`results.json`, JUnit) and `integration-artifacts`.
3. The category says where to look: `assertion` is the system under test, `threshold` is model quality against its gate, `infrastructure`/`timeout` is a service or the environment, `configuration`/`test_data` is the test's input.
4. Artifacts are under `artifacts/<run_id>/<module>/`: `api/*.json` (sanitized request and response), `database/queries*.json` (statements, timings, transactions), `ui/<test>/screenshot-*.png` and `diagnostics.json`, `ml/` and `llm_evaluation/` evaluation files.
5. Locally, rerun one test with `-x -vv`, and with `CAUTO_LOGGING__LEVEL=DEBUG` for framework logs.

## Isolation and parallel execution

Tests do not depend on order: each cross-module test has its own application and database, API test services reset per test, browser contexts are per test.

**Parallel runs.** pytest-xdist is not a project dependency; bring it in for a run:

```bash
uv run --with pytest-xdist pytest -n 4 tests/unit
CAUTO_EXECUTION__RUN_ID=my-run uv run --with pytest-xdist pytest -n 4 tests/integration tests/examples --cauto-report
```

The unit suite and the local integration and example suites pass with `-n 4` (Phase 8, Windows). Each worker writes its own `results-gw<N>.json` and the controller writes none; without `CAUTO_EXECUTION__RUN_ID` every worker gets its own run ID and folder. `python -m complete_automation.core.summary "reports/my-run/results*.json"` combines the worker files. CI runs serially.

## Extending the framework

**A new testing module.** Add `src/complete_automation/<module>/` that imports only the core; read its settings with `framework_settings.section("<module>", Model)`; record metrics, metadata and artifacts through `result_recorder` and `artifact_store`; subclass `FrameworkError` with the right `ErrorCategory`; ship a `pytest_plugin.py`. Add `tests/unit/<module>/` with a `test_architecture.py` like the others, add the module to `DOMAINS` in `tests/unit/test_module_independence.py`, declare its marker in `pyproject.toml`, mark its tests `pytestmark = [pytest.mark.<module>, pytest.mark.cauto_module("<module>")]`, and add its extra to the `uv sync` in CI if it needs one.

**A new cross-module workflow.** Put it in `tests/integration/cross_module/`, mark it `cross_module` plus each module it uses (and `ui` if it starts a browser), use the existing fixtures (`users_app`, `users_db_session`, `api_client_factory`, `ui_page`), generate data with `DataFactory`, verify through each module's own assertions, and keep database verification read-only. Extend `users_app.py` only with what the workflow needs, and document new endpoints in its docstring.

## Known limitations

- **CI on GitHub has not run this pipeline yet.** It was validated with actionlint and by running every job's commands on Linux (Ubuntu 26.04, Python 3.12) and Windows (Python 3.14). The previous pipeline has failed on `main` since Phase 4 in its integration step; the same command passes in every local reproduction, so the cause could not be confirmed. The new jobs publish JUnit, the JSON report, artifacts and a summary so the next failure can be read without the raw log.
- Real Ollama, PostgreSQL, live API and DeepEval tests are skipped in CI, not run.
- A missing prerequisite is reported as a skip with its reason; there is no separate "blocked" status in the result contract.
- An LLM judge failure is reported as `error` with category `unknown`, because `EvaluatorError` declares no more specific category.
- The users application is a test fixture: no authentication, single writer (SQLite).
- Parallel execution (pytest-xdist) is verified locally for the unit, integration and example suites, not in CI; real-model and PostgreSQL tests were not run in parallel.
