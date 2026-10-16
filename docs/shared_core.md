# Shared core

`complete_automation.core` is the domain-independent infrastructure every testing module uses: configuration, logging and redaction, exceptions, execution context, the test result contract, reporting, artifact storage, test data utilities, retry, assertion helpers and pytest fixtures.

**Boundaries.** The core has no HTTP client, database access, browser driver, ML or LLM code. It never imports a testing module. Domain modules (`api`, `db`, `ui`, `ml`, `llm`) import the core, never the other way round. `tests/unit/core/test_architecture.py` enforces this.

## Package structure

| Module | Owns |
| --- | --- |
| `core/config.py` | Typed settings, the loader and its precedence, per-module sections |
| `core/log.py` | Logging setup, execution identifiers on records, structured fields |
| `core/redaction.py` | Masking of sensitive keys and credential-shaped text |
| `core/exceptions.py` | `FrameworkError` hierarchy, `ErrorCategory`, `classify_exception` |
| `core/context.py` | `RunContext`, `TestContext`, current-context tracking |
| `core/results.py` | `TestResult`, `Status`, `MetricResult`, `FailureInfo`, `AttemptInfo` |
| `core/reporting.py` | `Reporter` protocol, `JsonReporter`, `summarize` |
| `core/artifacts.py` | `ArtifactStore` protocol, `LocalArtifactStore`, `ArtifactRef` |
| `core/datasets.py` | JSON, YAML, CSV and xlsx loaders, `DataLoader`, `validate_records`, `DataFactory` |
| `core/retry.py` | `RetryPolicy`, `call_with_retry`, `retrying` |
| `core/assertions.py` | `assert_keys_present`, `assert_subset` |
| `core/paths.py`, `core/serialization.py` | Safe path joining, explicit JSON conversion |
| `core/pytest_plugin.py` | Shared fixtures, `--cauto-*` options, result-reporting hook |

Importing any core module has no side effects: no logging configuration, no file reads, no network access.

## Installation

Python 3.12 or later, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run pytest tests/unit          # shared-core and other offline tests
```

The core's own dependencies are `pydantic` (typed models and validation), `PyYAML` (safe YAML loading), `python-dotenv` (`.env` files) and `pytest`.

## Configuration

```python
from complete_automation.core.config import load_settings

settings = load_settings()                      # environment from CAUTO_ENVIRONMENT, default "dev"
settings = load_settings(environment="test")    # explicit environment
settings = load_settings(config_dir="/path/to/config", env_file=False)
```

**Precedence**, lowest to highest:

1. Model defaults.
2. `config/environments/<environment>.yaml`.
3. `CAUTO_*` values in `.env`, next to the `config/` directory.
4. `CAUTO_*` process environment variables.
5. `overrides=` passed to `load_settings`.

**Environment selection:** the `environment=` argument, then `overrides["environment"]`, then `CAUTO_ENVIRONMENT` (process, then `.env`), then `dev`. An explicitly selected environment must have a YAML file. The default `dev` may run without one.

**Variable names:** `CAUTO_` plus the path with `__` between levels. For example, `CAUTO_LOGGING__LEVEL=DEBUG` and `CAUTO_MODULES__API__BASE_URL=https://...`. Values starting with `[` or `{` are parsed as JSON. Unknown keys are errors, not ignored. That also applies to stray `CAUTO_*` variables.

**Config directory:** `config_dir=`, else `$CAUTO_CONFIG_DIR`, else `<nearest pyproject.toml>/config`, else `./config`. Relative paths in settings (`reporting.output_dir`, `artifacts.root`, `logging.file`) resolve against the parent of the config directory, so results do not depend on the working directory.

**Sections:** `logging`, `reporting`, `artifacts`, `execution`, `retry`, `security`, and `modules.<name>` for domain modules.

| Key | Default | Meaning |
| --- | --- | --- |
| `logging.level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `logging.format` | `text` | `text` or `json` (one JSON object per line) |
| `logging.console` / `logging.file` | `true` / none | Outputs |
| `reporting.output_dir` | `reports` | JSON reports go to `<output_dir>/<run_id>/results.json` |
| `reporting.json_enabled` | `false` | Write the JSON report (or pass `--cauto-report`) |
| `artifacts.root` | `artifacts` | Artifacts go to `<root>/<run_id>/` |
| `execution.run_id` | generated | Fix the run ID, for example to share one across parallel workers |
| `retry.*` | 3 attempts, 0.5 s, ×2, cap 10 s, no jitter | Defaults for `RetryPolicy.from_settings` |
| `security.redact_keys` | `[]` | Extra key names to mask |
| `security.redaction_mask` | `***` | Replacement text |

**Errors:** every problem raises `ConfigurationError` naming the field (`retry.max_attempts: Input should be a valid integer`). Messages never include the rejected value, because that value may be a secret.

### Module sections

A domain module declares its own model and reads it through the same loader. There is no second loader, and no change to the core:

```python
from pydantic import BaseModel, SecretStr

class ApiSettings(BaseModel):
    base_url: str
    timeout_seconds: float = 10.0
    api_key: SecretStr | None = None

api = settings.section("api", ApiSettings)   # validates modules.api; ConfigurationError if invalid
```

## Secrets and redaction

- Never commit secrets. Put them in environment variables or a local `.env`, which git ignores. `.env.example` lists names only.
- Use `SecretStr` for secret fields in module sections. `repr(settings)` and `settings.redacted_dump()` mask sensitive keys, and so do log records and JSON reports.
- `Redactor` masks:
  - values under sensitive keys (`password`, `secret`, `token`, `api_key`, `authorization`, `cookie`, `credential`, `client_secret`, `connection_string`, ...) plus `security.redact_keys`;
  - credentials in URLs (`scheme://user:***@host`);
  - `Authorization` headers and bearer tokens;
  - `key=value` and `"key": "value"` pairs with sensitive keys;
  - PEM private keys.
- Look-alike keys such as `prompt_tokens`, `max_tokens` and `author` are kept.

**Limits:** redaction is pattern-based. A secret stored under an innocent key, in an unrecognised shape, is not detected. Modules must not log request bodies, datasets, model responses or artifacts by default. If redaction itself fails, the log record is withheld, and the raw text is never written.

## Logging

```python
from complete_automation.core.log import configure_logging, get_logger, log_context

configure_logging(settings.logging, redactor=settings.redactor())   # safe to call again
log = get_logger("api.client")                                       # complete_automation.api.client
with log_context(endpoint="/users"):
    log.info("request sent", extra={"status": 200})
```

- **Scope:** only the `complete_automation` logger is configured, never the root logger. Records still propagate, so pytest's `caplog` sees them.
- **Repeated calls:** calling `configure_logging` again replaces its earlier handlers instead of duplicating them. `reset_logging()` removes them.
- **Record contents:** every record carries `run_id`, `test_id` and `correlation_id` from the current context (`-` when there is none), plus fields from `log_context`.
- **Formats:** text format looks like `2026-10-16T09:00:00.000+00:00 INFO complete_automation.api.client [run=... test=... corr=...] request sent {"endpoint": "/users", "status": 200}`. JSON format writes the same fields as one object per line.

## Exceptions and error classification

`FrameworkError` is the base class. `ConfigurationError`, `TestDataError`, `ArtifactError`, `ReportingError` and `RetryExhaustedError` derive from it. Each class carries an `ErrorCategory`, and domain modules subclass `FrameworkError` with their own category:

```python
class ServiceUnavailable(FrameworkError):
    category = ErrorCategory.INFRASTRUCTURE
```

`classify_exception(error)` returns a category for any exception:

| Exception | Category |
| --- | --- |
| A `FrameworkError` | Its own category |
| `AssertionError` | `assertion` |
| `TimeoutError` | `timeout` |
| `OSError`, including connection errors | `infrastructure` |
| Anything else | `unknown` |

The remaining categories are `threshold`, `configuration` and `test_data`.

| Failure | Retry? | Treat as |
| --- | --- | --- |
| Configuration or test data error | Never: fix the input | Error |
| Artifact or reporting write failure | No | Infrastructure error, surfaced at once |
| Transient dependency failure (timeout, connection reset) | Only if the operation is safe to repeat, and only via an explicit `RetryPolicy` | Infrastructure error once retries are exhausted |
| Assertion or threshold failure | Never | Test failure |

## Execution context

```python
from complete_automation.core.context import RunContext, TestContext, use_test_context

run = RunContext.start("dev", suite="smoke")          # run_id like 20261016T090000Z-1a2b3c4d
test = TestContext(run_id=run.run_id, test_id="tests/api/test_users.py::test_list",
                   name="test_list", module="api")    # correlation_id generated
with use_test_context(test):
    ...                                                # logs carry test_id and correlation_id
finished = run.finish()                                # copy with finished_at set
```

- **Run-level vs test-level:** `RunContext` describes the whole run; `TestContext` describes one test.
- **Immutability:** both are immutable, and metadata is copied on creation.
- **Isolation:** the *current* context is tracked with `contextvars`, so each thread and asyncio task sees its own value. A new `threading.Thread` starts with no context. Under pytest-xdist, each worker is a separate process.
- **Contents:** contexts hold identifiers and plain metadata only, never clients, sessions or connections.

## Test result contract

`TestResult` (schema version `1.0`) is one test's outcome in any discipline:

| Field | Type | Notes |
| --- | --- | --- |
| `run_id`, `test_id`, `name` | str | `test_id` is the pytest node ID |
| `module` | str | Lower-case discipline or suite, for example `api` or `llm` |
| `status` | `Status` | See below |
| `started_at`, `finished_at` | aware datetime | `duration_seconds` is computed |
| `correlation_id` | str or none | |
| `failure` | `FailureInfo` or none | `message`, `category`, `exception_type`, `details` (redacted) |
| `metrics` | tuple of `MetricResult` | `name`, `value`, `threshold`, `passed`, `unit`, `metadata` |
| `artifacts` | tuple of `ArtifactRef` | |
| `metadata` | dict of JSON values | Non-sensitive; non-JSON values are rejected, never dropped |
| `attempt` | `AttemptInfo` or none | `attempt` ≤ `max_attempts` |
| `tags` | tuple of str | Marker names, from the plugin |

**Statuses:**

- **`passed`:** every check passed.
- **`failed`:** a check failed, or a metric missed its threshold. The system under test is at fault.
- **`error`:** the test could not run properly. Causes include setup, infrastructure, configuration, test data, or a defect in the test.
- **`skipped`:** deliberately not run. This includes `xfail`.
- **`incomplete`:** the test ran, but no verdict was possible (for example, a judge returned no usable score).

**Validation rules:**

- `failed` and `error` require `failure`; `passed` and `skipped` forbid it.
- Unknown fields, naive timestamps, non-finite metric values and non-JSON metadata are rejected.

**Serialization:** `result.to_dict()` gives JSON-compatible data, with timestamps as ISO 8601 UTC (`...Z`). `TestResult.from_dict(data)` rebuilds the result.

## Reporting

```python
from complete_automation.core.reporting import JsonReporter

reporter = JsonReporter(settings.reporting.output_dir / run.run_id / "results.json",
                        redactor=settings.redactor())
reporter.start_run(run)
reporter.publish(result)          # any number of times
path = reporter.finalize(run.finish())
```

`Reporter` is a protocol with `start_run(run)`, `publish(result)` and `finalize(run=None) -> Path | None`. Future HTML, JUnit or CI reporters implement the same three methods.

The JSON document is written atomically, with sorted keys:

```json
{
  "schema_version": "1.0",
  "run": {"run_id": "...", "environment": "test", "started_at": "...", "finished_at": "...", "suite": null, "metadata": {}},
  "summary": {"total": 5, "passed": 1, "failed": 1, "skipped": 1, "error": 1, "incomplete": 1,
              "duration_seconds": 4.5, "artifact_count": 0, "metric_count": 1,
              "by_module": {"api": {"total": 2, "passed": 1, "failed": 1, "skipped": 0, "error": 0, "incomplete": 0}}},
  "results": [ ... TestResult.to_dict() ... ]
}
```

**What the reporter does not do:** it counts statuses and passes metrics through unchanged. It never recomputes them.

**Failure behaviour:** publishing after `finalize`, a non-`TestResult` input, or a write failure raises `ReportingError`. In a pytest run, that error makes the session fail.

## Artifacts

```python
from complete_automation.core.artifacts import LocalArtifactStore

store = LocalArtifactStore.for_run(settings.artifacts.root, run.run_id)
ref = store.write_json("evaluation.json", {"score": 0.82}, subdir="llm/test_answer")
store.write_text("trace.txt", "...")
store.write_bytes("screen.png", png_bytes, media_type="image/png")
store.resolve(ref)   # absolute path, refuses references outside the store
```

**Write behaviour:**

- Names are sanitised to `[A-Za-z0-9._-]`.
- `subdir` must stay inside the store: `..` and absolute paths raise `ArtifactError`.
- Files are created exclusively. A repeated name becomes `name-1.ext`, `name-2.ext`, and existing files are never overwritten.
- The directory is created on first write.

**`ArtifactRef`** records `name`, `path` (POSIX, relative to the store), `media_type`, `size_bytes`, `sha256`, `created_at` and `metadata`. Attach it to a result. Contents are stored as given, so redact sensitive diagnostics before writing them.

`ArtifactStore` is a protocol, so an object-storage adapter can replace `LocalArtifactStore` without changing testing modules.

## Pytest fixtures and lifecycle

`tests/conftest.py` enables the plugin:

```python
pytest_plugins = ["complete_automation.core.pytest_plugin", "pytester"]
```

| Fixture | Scope | Provides |
| --- | --- | --- |
| `framework_settings` | session | Validated `FrameworkSettings`, loaded once at start-up |
| `run_context` | session | `RunContext` for this run |
| `framework_logger` | session | `complete_automation.tests` logger |
| `artifact_store` | session | `LocalArtifactStore` at `<artifacts.root>/<run_id>` |
| `test_context` | function | `TestContext`, current for the test; reset afterwards even when the test fails |
| `result_recorder` | function | `add_metric`, `attach`, `add_metadata`, `mark_incomplete` for this test's result |

**Options:** `--cauto-env NAME`, `--cauto-config-dir PATH`, and `--cauto-report` (equivalent to `reporting.json_enabled: true`).

**Fixture rules:**

- No fixture is autouse, and none makes network calls.
- Invalid configuration stops pytest with a usage error (exit code 4).

**Reporting hook:** when reporting is enabled, each test produces exactly one `TestResult`, published after its teardown. The report path is printed in the terminal summary. Mapping from pytest outcomes:

- A setup failure is `error`. A teardown failure turns a passed or skipped test into `error`.
- `AssertionError` is `failed`, with category `assertion`.
- `AssertionError` together with a recorded metric where `passed=False` is `failed`, with category `threshold`.
- Any other exception is `error`, categorised by `classify_exception`.
- `pytest.skip` and `xfail` are `skipped`, with the reason in `metadata.skip_reason`. A non-strict XPASS is `passed`, with `metadata.xpass`.
- `mark_incomplete(reason)` on an otherwise passing test gives `incomplete`.

**Module and tags:** `@pytest.mark.cauto_module("api")` sets `module`; otherwise the folder under `tests/` is used. Other marker names become `tags`.

**Parallel runs:** under pytest-xdist, each worker writes `results-<worker>.json`. Set `CAUTO_EXECUTION__RUN_ID` to put all workers in one run folder. Merging worker files is not implemented.

## Test data

```python
from complete_automation.core.datasets import DataFactory, DataLoader

loader = DataLoader(settings.base_dir / "data")   # loading is confined to this root
cases = loader.load("llm/eval_cases.json")        # .json, .yaml/.yml, .csv, .xlsx
users = loader.load_records("api/users.yaml", UserModel)   # validated with a pydantic model
```

**Loader behaviour:**

- Paths outside the root, unknown extensions, missing files and malformed content raise `TestDataError`.
- CSV rows with the wrong number of fields are rejected.
- xlsx reading needs no extra dependency. It reads the first worksheet and keeps blank cells in place.
- Validation errors name the record and field, not the value.

**`DataFactory(seed)`** is reproducible. It offers `integer`, `string`, `email`, `choice`, `uuid` and `sequence("user")`, which yields `user-0001`.

## Retry

```python
from complete_automation.core.retry import RetryPolicy, call_with_retry, retrying

policy = RetryPolicy.from_settings(settings.retry, retry_on=(ConnectionError, TimeoutError))
data = call_with_retry(lambda: fetch_status(), policy)

@retrying(policy)
def fetch_status(): ...
```

- **Explicit opt-in:** nothing is retried unless listed. `Exception`, `BaseException` and `AssertionError` are refused as retry targets.
- **Unlisted errors** propagate immediately, unwrapped.
- **Exhaustion** raises `RetryExhaustedError`, with the last error chained.
- **Delays** grow by `backoff_multiplier`, are capped at `max_delay_seconds`, and can be jittered by `±jitter_ratio`.
- **Safety:** only retry operations that are safe to repeat.
- **Testing:** `sleep=` and `rng=` are injectable, for fast, deterministic tests.

## Assertion helpers

`assert_keys_present(data, ["id", "name"], label="user")` and `assert_subset(actual, expected, label="body")` report every missing key or mismatching path at once, with short, redacted values. Use a plain `assert` for everything else.

## Extending the core from a module

A future API module, for example:

```python
# complete_automation/api/settings.py
class ApiSettings(BaseModel):
    base_url: str
    api_key: SecretStr

# tests/api/conftest.py
@pytest.fixture(scope="session")
def api_settings(framework_settings):
    return framework_settings.section("api", ApiSettings)

# tests/api/test_users.py
@pytest.mark.cauto_module("api")
def test_list_users(api_client, artifact_store, result_recorder):
    response = api_client.get("/users")
    result_recorder.attach(artifact_store.write_json("response.json", response.json(), subdir="api"))
    result_recorder.add_metric("latency", response.elapsed_ms, unit="ms")
    assert response.status_code == 200
```

Changes to a shared contract (`TestResult`, `ArtifactRef`, `Reporter`, `ArtifactStore`, settings sections) must be documented, versioned (`schema_version`), tested in `tests/unit/core`, and agreed with every module that uses them.

## Migration notes (Phase 1)

| Change | Impact |
| --- | --- |
| `complete_automation.core.llm_app` moved to `complete_automation.llm.app` | LLM code no longer lives in the core. The only caller (`tests/llm/conftest.py`) is updated. External imports must change. |
| `utils.excel_reader.read_llm_test_data` now uses `core.datasets.read_xlsx_rows` | Same output for workbooks without blank cells. Blank cells no longer shift later columns left (a bug fix). |
| `config/settings.py` and `reporting/report_manager.py` | Unchanged and still used by the LLM suite. New code should use `core.config` and `core.artifacts` / `core.reporting`. |
| New `tests/conftest.py` loads the plugin | It adds fixtures and options only. The LLM suite's fixtures (`framework_setup`, `llm_test_cases`) are unchanged. |
| New dependencies | `pydantic` (already installed through deepeval, now declared) and `PyYAML`. |

## Examples

### Runnable end-to-end example

```python
from pathlib import Path
import tempfile

from complete_automation.core.config import load_settings
from complete_automation.core.context import RunContext, TestContext, use_test_context
from complete_automation.core.log import configure_logging, get_logger
from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.reporting import JsonReporter
from complete_automation.core.results import MetricResult, Status, TestResult
from complete_automation.core.context import utc_now

workdir = Path(tempfile.mkdtemp())
(workdir / "config" / "environments").mkdir(parents=True)

# 1. Load configuration (explicit directory, no .env, no inherited variables)
settings = load_settings(config_dir=workdir / "config", env_file=False, environ={})

# 2. Obtain a shared logger
configure_logging(settings.logging, redactor=settings.redactor())
log = get_logger("example")

# 3. Create an execution context
run = RunContext.start(settings.environment)
test = TestContext(run_id=run.run_id, test_id="example::test_answer", name="test_answer", module="llm")

with use_test_context(test):
    log.info("evaluating with api_key=%s", "not-logged")

    # 6. Store an artifact
    store = LocalArtifactStore.for_run(settings.artifacts.root, run.run_id)
    ref = store.write_json("evaluation.json", {"faithfulness": 0.82}, subdir="llm")

    # 4. Create and serialize a common test result with the artifact attached
    started = test.started_at
    result = TestResult(
        run_id=run.run_id, test_id=test.test_id, name=test.name, module=test.module,
        status=Status.PASSED, started_at=started, finished_at=utc_now(),
        correlation_id=test.correlation_id,
        metrics=(MetricResult(name="faithfulness", value=0.82, threshold=0.7, passed=True),),
        artifacts=(ref,),
    )
    print(result.to_dict()["status"], result.to_dict()["artifacts"][0]["path"])

# 5. Publish the result to the reporter
reporter = JsonReporter(settings.reporting.output_dir / run.run_id / "results.json")
reporter.start_run(run)
reporter.publish(result)
print(reporter.finalize(run.finish()).name)
```

Expected output: the log line with `api_key=***`, then `passed llm/evaluation.json`, then `results.json`.

### 7. Running the shared-core tests

```bash
uv run pytest tests/unit/core          # all shared-core unit tests
uv run pytest tests/unit               # plus the dataset tests
uv run ruff check src tests
```
