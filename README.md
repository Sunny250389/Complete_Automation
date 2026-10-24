# Complete_Automation

A Python test automation framework built on pytest. It covers UI, API, database, ML model evaluation and LLM evaluation, with one configuration system, one result format and one report per run.

This README is the starting point. Each module's full reference is in `docs/`, linked from the sections below.

## Contents

1. [Install](#1-install)
2. [Repository layout](#2-repository-layout)
3. [Your first test: a login test for your application (UI)](#3-your-first-test-a-login-test-for-your-application-ui)
4. [Where test data goes](#4-where-test-data-goes)
5. [API tests](#5-api-tests)
6. [Database tests](#6-database-tests)
7. [LLM model tests](#7-llm-model-tests)
8. [ML model tests](#8-ml-model-tests)
9. [Running tests](#9-running-tests)
10. [Configuration](#10-configuration)
11. [What is in the repository](#11-what-is-in-the-repository)

## 1. Install

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-extras        # every module; or --extra ui, --extra api, --extra database, --extra llm, --extra ml
uv run playwright install chromium   # browser for UI tests (or: firefox, webkit)
uv run pytest --collect-only         # collects every test without running it; a quick install check
```

Plain `uv sync --locked` installs the shared core only. Each module's extra is optional: without it, that module's tests are left out of collection. Windows note: `playwright install --with-deps` is for Linux; on Windows, `playwright install chromium` is enough.

## 2. Repository layout

| Folder | What goes there |
| --- | --- |
| `src/complete_automation/` | The framework. You normally do not change it to write tests. |
| `tests/` | **Your tests.** One folder per application or area (see section 3). |
| `tests/examples/` | Runnable examples, one per module. Copy from them. |
| `tests/support/` | Helpers the examples use: a local practice site, a local API service, page objects. |
| `data/` | **Test data**, one folder per module (see section 4). |
| `config/environments/` | One YAML file per environment: `dev.yaml` (default), `test.yaml`, and examples such as `ui_local.yaml`. |
| `reports/`, `artifacts/` | Generated: JSON reports and evidence (screenshots, exchanges, evaluations). Git-ignored. |
| `docs/` | Full reference for each module. |

## 3. Your first test: a login test for your application (UI)

This walks through testing a login page of your own web application with Playwright, through pytest. The pieces are: a **configuration file** that gives the application's address, a **page object** that knows the login page's elements, and a **test file** that uses the `ui_page` and `ui_base_url` fixtures.

### Step 1: describe the environment

Create `config/environments/myapp.yaml`. The file name (without `.yaml`) is the environment name you pass to pytest.

```yaml
# config/environments/myapp.yaml
modules:
  ui:
    base_url: https://myapp.example.com   # or http://127.0.0.1:3000 for a local application
    browser: chromium                      # chromium, firefox or webkit
    headless: false                        # set to true on CI; false shows the browser while you write the test
    screenshot: on_failure                 # a screenshot is kept automatically when a test fails
```

Do not put passwords in this file. Credentials come from environment variables (step 3).

### Step 2: write a page object

A page object holds the locators and the actions for one screen, so the test reads as intent. Put it in `tests/myapp/pages/`.

```python
# tests/myapp/pages/login_page.py
from playwright.sync_api import Page

from complete_automation.ui.pages import BasePage, FormField, Notice


class LoginPage(BasePage):
    path = "/login"  # appended to modules.ui.base_url

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        # Find controls by their accessible label. Error messages are found by a test id.
        self.email = FormField(page, "Email", error_test_id="email-error")
        self.password = FormField(page, "Password", error_test_id="password-error")
        self.submit_button = self.by_role("button", "Sign in")
        self.error = Notice(page, "login-error")

    def sign_in(self, email: str, password: str) -> None:
        self.email.fill(email)
        self.password.fill(password)
        self.click(self.submit_button)
```

`BasePage` gives you `open()`, `by_role()`, `by_label()`, `by_test_id()`, `by_text()`, `click()`, `fill()`, `select()` and `set_checked()`. Prefer `by_role` and `by_label`; add a `data-testid` attribute to an element in your application only when no accessible name works.

### Step 3: write the test

Put the test in `tests/myapp/ui/test_login.py`. The `ui_page` fixture gives a fresh browser page for each test, and `ui_base_url` gives the address from the configuration. You do not start or stop the browser yourself.

```python
# tests/myapp/ui/test_login.py
import os
import re

import pytest
from playwright.sync_api import Page

from complete_automation.ui.assertions import assert_text, assert_url, assert_visible
from tests.myapp.pages.login_page import LoginPage

pytestmark = [pytest.mark.ui, pytest.mark.cauto_module("ui")]


@pytest.fixture
def test_user() -> tuple[str, str]:
    """Credentials from the environment, never from the repository."""
    email = os.environ.get("MYAPP_TEST_EMAIL")
    password = os.environ.get("MYAPP_TEST_PASSWORD")
    if not email or not password:
        pytest.skip("set MYAPP_TEST_EMAIL and MYAPP_TEST_PASSWORD to run the login tests")
    return email, password


def test_valid_login_opens_the_dashboard(ui_page: Page, ui_base_url: str, test_user: tuple[str, str]) -> None:
    login = LoginPage(ui_page, ui_base_url)
    login.open()

    login.sign_in(*test_user)

    assert_url(ui_page, re.compile(r"/dashboard$"))


def test_wrong_password_shows_an_error(ui_page: Page, ui_base_url: str, test_user: tuple[str, str]) -> None:
    login = LoginPage(ui_page, ui_base_url)
    login.open()

    login.sign_in(test_user[0], "definitely-wrong")

    assert_visible(login.error.locator)
    assert_text(login.error.locator, "Email or password is incorrect")  # substring match by default
    assert_url(ui_page, re.compile(r"/login$"))


def test_empty_form_shows_validation_messages(ui_page: Page, ui_base_url: str) -> None:
    login = LoginPage(ui_page, ui_base_url)
    login.open()

    login.sign_in("", "")

    assert_visible(login.email.error)
    assert_visible(login.password.error)
```

Two points about the file:

- `tests/myapp/` needs an empty `__init__.py` in each folder, as the existing `tests/` folders have.
- Do **not** import `tests.support.ui_site` in your tests. That module points the UI settings at the bundled practice site. Your tests should use the `ui_settings` from the plugin, which reads `modules.ui` from your environment file.

### Step 4: run it

```bash
export MYAPP_TEST_EMAIL=you@example.com          # PowerShell: $env:MYAPP_TEST_EMAIL = "you@example.com"
export MYAPP_TEST_PASSWORD='your-password'
uv run pytest tests/myapp/ui --cauto-env myapp -v
```

On failure, a screenshot of the page is saved under `artifacts/`. Add `--cauto-report` to also write `reports/<run_id>/results.json`.

### See it working without your own application

`tests/examples/ui/test_login_examples.py` runs the same pattern against the bundled practice site (`tests/support/ui_site/login.html`), with no setup:

```bash
uv run pytest tests/examples/ui -v
```

The practice site also has `index.html` (navigation, a delayed element) and `form.html` (validation, radio buttons, dropdowns); see `tests/examples/ui/test_ui_examples.py`.

### UI checklist

- Start from a page object. Keep locators and actions out of the test file.
- Use the assertions in `complete_automation.ui.assertions` (`assert_visible`, `assert_text`, `assert_url`, ...). They wait automatically, so do not add `sleep` calls.
- Keep credentials in environment variables; read them with `os.environ`.
- Set `headless: true` on CI, and `trace: on_failure` only when you need one (traces can contain cookies).

## 4. Where test data goes

Each module reads its data from its own folder under `data/`. Loading is confined to that folder: a path outside it is rejected.

| Test kind | Put the data in | How the test reads it |
| --- | --- | --- |
| UI | `data/ui/` (for example `data/ui/login_users.json`) | `DataLoader(<repo>/data).load("ui/login_users.json")`, as in `tests/examples/ui/test_login_examples.py` |
| API | `data/api/` (for example `data/api/contracts/users_api.openapi.yaml` for a contract) | `OpenApiContract.from_file(...)`, or the `openapi_contract` fixture |
| Database | `data/database/` by convention (seed CSV or JSON files) | `DataLoader` in the test; the examples create their data in code |
| LLM evaluation | `data/llm_evaluation/<your-folder>/` (for example `data/llm_evaluation/myapp/summaries.jsonl`) | the `llm_load_dataset` fixture, relative to `data/llm_evaluation/` |
| LLM baselines | `data/llm_evaluation/baselines/` | the `llm_load_baseline` fixture |
| ML evaluation | `data/ml/` (datasets and `data/ml/baselines/`) | the `ml_dataset_loader` and `ml_load_baseline` fixtures, relative to `data/ml/` |
| Anything else | anywhere under `data/` | `DataLoader(<repo>/data)` from `complete_automation.core.datasets` |

Supported formats: JSON, YAML (safe loader only), CSV, `.xlsx` (standard library, no extra dependency) and JSONL for LLM datasets. Validation errors name the record and the field, not the value.

Rules for test data:

- **No real customer data and no real credentials** in `data/`. Put credentials in environment variables.
- Keep test data small and reviewable. Generate large or random data in the test with `DataFactory(seed)` from `complete_automation.core.datasets`.
- The practice-site credentials in `data/ui/login_users.json` are fake and only work on the bundled practice site.

## 5. API tests

The API module provides HTTPX clients (sync and async), authentication, assertions, JSON Schema and OpenAPI contract checks, retries and mocks. Full reference: [docs/api_automation.md](docs/api_automation.md).

### Step 1: configure the service

```yaml
# config/environments/myapp.yaml (add to the same file as the UI settings, or use a new file)
modules:
  api:
    base_url: https://api.myapp.example.com/v1
    timeouts: {connect: 5, read: 30, write: 30, pool: 5}
    auth: {type: bearer}          # the token comes from CAUTO_MODULES__API__AUTH__TOKEN
```

Other auth types are `api_key`, `basic` and `oauth2_client_credentials`; see the table in `docs/api_automation.md`.

### Step 2: write the test

```python
# tests/myapp/api/conftest.py
from complete_automation.api.pytest_plugin import *  # noqa: F401,F403  (registers the api fixtures here)
```

```python
# tests/myapp/api/test_users.py
import pytest

from complete_automation.api.assertions import assert_field, assert_json_keys, assert_latency, assert_status

pytestmark = [pytest.mark.api, pytest.mark.cauto_module("api")]


def test_get_user_returns_the_profile(api_client) -> None:
    response = api_client.get("/users/{user_id}", path_params={"user_id": 1})

    assert_status(response, 200)
    assert_json_keys(response, ["id", "email", "role"])
    assert_field(response, "id", equals=1)
    assert_latency(response, 2000)  # milliseconds


def test_missing_user_returns_404(api_client) -> None:
    response = api_client.get("/users/{user_id}", path_params={"user_id": 999999})

    assert_status(response, 404)
```

- `api_client` is a client already configured from `modules.api`. Use `api_client_factory(base_url=..., auth=...)` when one test needs a different service or credentials.
- Validate a response against an OpenAPI contract with the `openapi_contract` fixture, or `CONTRACT.assert_response(response)` as in the example.

### Try it

```bash
uv run pytest tests/examples/api -v
```

`tests/examples/api/test_users_api_examples.py` runs against a local test service that the examples start on `127.0.0.1`. It shows authentication failures, validation errors, schema mismatches and contract violations. Its `conftest.py` also starts that local service; do not copy that part into your own tests.

## 6. Database tests

The database module wraps SQLAlchemy with safe parameterized SQL, transactions, read-only access by default, schema validation, reconciliation and fixtures. Full reference: [docs/database_testing.md](docs/database_testing.md).

**Safety defaults.** A configured database is **read-only** unless you set `access: read_write`. DDL needs `allow_ddl: true`, and destructive statements need `allow_destructive: true`. Environments named `prod` or `production` are protected by default. Point tests at a test database, never at production.

### Step 1: configure the database

```yaml
# config/environments/myapp.yaml
modules:
  database:
    dialect: postgresql           # or sqlite
    host: db.myapp.example.test
    port: 5432
    database: app_test
    username: app_tester          # the password comes from CAUTO_MODULES__DATABASE__PASSWORD
    schema: app
    access: read_only             # the default; change only for a test database you own
```

For SQLite, use `dialect: sqlite` and `database: path/to/file.db`.

### Step 2: write the test

```python
# tests/myapp/database/conftest.py
from complete_automation.database.pytest_plugin import *  # noqa: F401,F403
```

```python
# tests/myapp/database/test_orders.py
import pytest

from complete_automation.database.validations import assert_record, assert_row_count

pytestmark = [pytest.mark.database, pytest.mark.cauto_module("database")]


def test_every_order_has_a_status(db_session) -> None:
    assert db_session.scalar("SELECT count(*) FROM orders WHERE status IS NULL") == 0


def test_a_known_order_is_stored(db_session) -> None:
    assert_record(db_session, "orders", {"id": 1001}, {"status": "paid", "customer_id": 42})


def test_order_count_for_a_customer(db_session) -> None:
    assert_row_count(db_session, "orders", 3, {"customer_id": 42})
```

- `db_session` gives one session on the configured database and cleans it up after the test. It skips with the reason if the database is not configured.
- Use bound parameters (`db_session.query("... WHERE id = :id", {"id": 1})`). Never build SQL by string formatting.
- To check a change and undo it, use `db_session.execute(...)` followed by `db_session.rollback()`, as in example 12 of `tests/examples/database/test_database_examples.py`.

### Try it

```bash
uv run pytest tests/examples/database -v        # runs on temporary SQLite databases; PostgreSQL examples skip
uv run pytest -m postgresql --cauto-env postgres_test   # the PostgreSQL examples, with a configured server
```

## 7. LLM model tests

The LLM module calls a real, locally installed [Ollama](https://ollama.com) model, captures the response it generated, and evaluates it with deterministic checks, an optional judge model, quality gates and baselines. Full reference: [docs/llm_evaluation.md](docs/llm_evaluation.md).

**Opt-in.** Real-model tests skip with a reason unless you set the model and enable live tests. Nothing in the module pulls or downloads a model; install the model yourself with `ollama pull <name>`.

### Step 1: install and pick a model

```bash
ollama list                              # the names of installed models
```

### Step 2: configure the model

`config/environments/ollama_local.yaml` already turns live tests on and sets a 300 s timeout. Name the model in your shell, not in the file:

```bash
export CAUTO_MODULES__LLM_EVALUATION__MODEL=qwen3:4b       # a name from `ollama list`
export CAUTO_MODULES__LLM_EVALUATION__JUDGE_MODEL=qwen3:4b  # optional; may be a different model
```

PowerShell: `$env:CAUTO_MODULES__LLM_EVALUATION__MODEL = "qwen3:4b"`.

### Step 3: write a test

Put your cases in a JSONL file under `data/llm_evaluation/`, one case per line:

```json
{"case_id": "summary-length", "prompt": "Summarise: The office moves on Monday.", "checks": [{"type": "not_empty"}, {"type": "completed"}, {"type": "length", "maximum": 40, "unit": "words"}], "tags": ["summary"]}
```

Then write the test:

```python
# tests/myapp/llm/conftest.py
from complete_automation.llm_evaluation.pytest_plugin import *  # noqa: F401,F403
```

```python
# tests/myapp/llm/test_summaries.py
import pytest

from complete_automation.llm_evaluation import EvaluationConfig, GenerationConfig

pytestmark = [pytest.mark.llm_evaluation, pytest.mark.llm_live, pytest.mark.cauto_module("llm_evaluation")]


def test_the_model_answers_the_summary_cases(llm_settings, llm_live_model, llm_load_dataset, llm_evaluate) -> None:
    dataset = llm_load_dataset("myapp/summaries.jsonl")       # validated before anything runs
    config = EvaluationConfig(model=llm_settings.target_model(GenerationConfig(temperature=0.0, seed=1, max_tokens=400)))

    report = llm_evaluate(dataset, config, dataset_id="myapp-summaries", assert_passed=True)

    assert report.passed
```

- `llm_evaluate` runs the model, evaluates each response, applies the gates and writes the artifacts (`artifacts/llm_evaluation/`). `assert_passed=True` raises if a check, judged criterion or threshold fails.
- Checks are deterministic: `not_empty`, `completed`, `contains`, `not_contains`, `regex`, `json_valid`, `json_schema`, `json_keys`, `length`, `refusal`, `exact_match`, `context_overlap`. The complete list and the caveats for `refusal` and `context_overlap` are in the doc.
- To grade a free-form answer, add a judge: `criteria` on a case, and a `judge` (`JudgeConfig`) on `EvaluationConfig`. The judge model can be set with `modules.llm_evaluation.judge_model`. Judge scores are estimates from a language model, not ground truth.
- Add a quality gate across the run: `suite_thresholds` on `EvaluationConfig`, for example `case_pass_rate >= 0.9`.

### Try it

```bash
uv run pytest tests/examples/llm -v                      # the offline example always runs; the live one skips without a model
uv run pytest tests/integration/llm_evaluation/test_ollama_live.py --cauto-env ollama_local -s -v
```

`tests/examples/llm/test_llm_examples.py` loads `data/llm_evaluation/examples/ollama_smoke.jsonl` and, when a model is configured, runs it against your model. The real-model tests were run against `qwen3:4b` on Ollama 0.40.8.

**Privacy.** Reports store prompts and responses by default. For sensitive prompts set `modules.llm_evaluation.persist_raw: false`.

## 8. ML model tests

The ML module evaluates classification, regression and clustering models, or precomputed predictions, against validated datasets, with explicit handling of undefined metrics, threshold gates and baselines. It never trains or serves a model. Full reference: [docs/ml_evaluation.md](docs/ml_evaluation.md).

### Write a test

Put the dataset in `data/ml/` (for example `data/ml/myapp/churn_predictions.csv`) with the features, the true label and a column of predicted probabilities or labels. Then:

```python
# tests/myapp/ml/conftest.py
from complete_automation.ml.pytest_plugin import *  # noqa: F401,F403
```

```python
# tests/myapp/ml/test_churn_model.py
import pytest

from complete_automation.ml import EvaluationConfig, PrecomputedAdapter, ThresholdRule

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]


def test_churn_predictions_meet_the_quality_gate(ml_dataset_loader, ml_evaluate) -> None:
    dataset = ml_dataset_loader.load(
        "myapp/churn_predictions.csv", features=["tenure", "monthly_spend"], target="churned",
        dataset_id="churn", version="1",
    )
    model = PrecomputedAdapter.from_frame(dataset.frame, "prediction", model_id="churn-v3", version="3")
    config = EvaluationConfig(task="classification", positive_label=1,
                              thresholds=(ThresholdRule.at_least("accuracy", 0.80),))

    ml_evaluate(dataset, model, config, assert_passed=True)
```

Use `EstimatorAdapter` to evaluate an in-memory scikit-learn model, and `ml_load_baseline` with `BaselineRule` to compare with a stored baseline in `data/ml/baselines/`.

### Try it

```bash
uv run pytest tests/examples/ml -v
```

`tests/examples/ml/test_ml_examples.py` is the reference for the loader, the adapters, thresholds and baselines. It needs the `ml` extra (numpy, pandas, scikit-learn).

## 9. Running tests

| Command | What it runs |
| --- | --- |
| `uv run pytest --collect-only` | Collects every test without running it |
| `uv run pytest tests/unit` | Offline unit tests for every module |
| `uv run pytest tests/examples` | The examples for every module |
| `uv run pytest tests/integration tests/examples` | Local integration tests: API service, SQLite, browser, cross-module |
| `uv run pytest -m ui` | Every UI test (needs the `ui` extra and an installed browser) |
| `uv run pytest -m api` | Every API test (live ones skip unless configured) |
| `uv run pytest -m database` | Every database test (PostgreSQL ones skip unless configured) |
| `uv run pytest -m ml` | ML evaluation tests |
| `uv run pytest -m llm_evaluation` | LLM evaluation tests; real-model ones skip unless opted in |
| `uv run pytest tests/myapp --cauto-env myapp` | Your tests, with the `myapp` environment |
| `uv run pytest ... --cauto-report` | Also writes `reports/<run_id>/results.json` |
| `uv run python -m complete_automation.core.summary "reports/<run_id>/results*.json"` | Markdown summary of a report |
| `uv run pytest` | Everything; tests that need a service skip with their reason |

Useful pytest options: `-v` (names), `-s` (print output), `-k <expression>` (select by name), `-x` (stop at the first failure), `-m "<marker expression>"` (select by marker).

Each test is recorded with its module (from the `cauto_module` marker, or the folder under `tests/`). Tests that skip say why in the report.

## 10. Configuration

Settings come from, in order (later wins):

1. `config/environments/<env>.yaml`, selected with `--cauto-env <env>` or `CAUTO_ENVIRONMENT`
2. `CAUTO_*` values in a local `.env` file (copy `.env.example`)
3. `CAUTO_*` environment variables

Nested keys use `__`: `CAUTO_MODULES__API__AUTH__TOKEN=...` sets `modules.api.auth.token`.

- **Secrets** (passwords, tokens, API keys) go in `.env` or environment variables, never in YAML. `.env` is git-ignored.
- **Credentials for your application's tests** use names you choose, for example `MYAPP_TEST_EMAIL`, and are read with `os.environ` in the test.
- The full rules are in [docs/shared_core.md](docs/shared_core.md#configuration).

## 11. What is in the repository

- `src/complete_automation/core/`: shared infrastructure: configuration, logging and redaction, the result contract, reporting, artifacts, test data loaders, retries and the pytest fixtures for all modules. See [docs/shared_core.md](docs/shared_core.md).
- `src/complete_automation/api/`: API automation. See [docs/api_automation.md](docs/api_automation.md).
- `src/complete_automation/database/`: database testing, SQLite and PostgreSQL. See [docs/database_testing.md](docs/database_testing.md).
- `src/complete_automation/ui/`: UI automation with Playwright. See [docs/ui_automation.md](docs/ui_automation.md).
- `src/complete_automation/ml/`: ML model evaluation. See [docs/ml_evaluation.md](docs/ml_evaluation.md).
- `src/complete_automation/llm_evaluation/`: LLM evaluation with Ollama. See [docs/llm_evaluation.md](docs/llm_evaluation.md).
- `tests/integration/`, `tests/examples/`: tests against local services: an API service on 127.0.0.1, temporary SQLite, a practice site in a headless browser, and cross-module workflows against a local users application (`tests/support/users_app.py`).
- `tests/live/`: opt-in tests against real external APIs (`reqres.yaml`); they skip unless configured.
- `.github/workflows/ci.yml`: static validation, offline unit tests and local integration tests, with reports and a job summary. See [docs/integration_and_ci.md](docs/integration_and_ci.md).
- `.github/workflows/live-llm.yml`: manual workflow that runs the real-model LLM tests against a pinned Ollama on the runner.
- `src/complete_automation/llm/`, `metrics/`, `utils/`, `config/`, `reporting/`, `tests/llm/`: the older DeepEval prototype. It evaluates the stub in `src/complete_automation/llm/app.py`, not a real model, and needs `OPENAI_API_KEY` for its judge; without that key its tests skip.
- `data/llm/eval_cases.json` and `eval_cases.xlsx`: the six-case seed dataset used by the DeepEval prototype and by the shared-core loader example. The two files hold the same rows.
- `RAG-Testing-Framework/`: a RAG prototype outside the package. Its tests are not run by the framework's pytest configuration; see the note in `RAG-Testing-Framework/tests/test_rag.py`.
