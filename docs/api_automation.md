# API automation

`complete_automation.api` tests HTTP APIs. It covers clients, authentication, request handling, response assertions, JSON Schema and OpenAPI contract validation, retries, mocking, and pytest fixtures.

Everything shared comes from [the shared core](shared_core.md): configuration, logging and redaction, results, reporting and artifacts. The API module adds no second version of any of them.

**Boundaries.**

- The module depends only on `complete_automation.core`, HTTPX, `jsonschema` and the OpenAPI validators.
- It contains no browser, database, ML or LLM code, and the core never imports it.
- `tests/unit/api/test_architecture.py` enforces both rules.
- Importing the package makes no network call.

## Installation

The module is an optional extra:

```bash
uv sync --locked --extra api        # or --all-extras, as CI does
```

The extra adds `httpx`, `jsonschema` and `openapi-spec-validator` (which brings `openapi-schema-validator`). Without it, the core still works, and the API test folders are left out of collection.

## Architecture

| Module | Public interface |
| --- | --- |
| `api/client.py` | `ApiClient`, `AsyncApiClient`, `ApiResponse`, `RetryConfig`, `AttemptRecord`, `parse_retry_after` |
| `api/auth.py` | `ApiKeyAuth`, `BearerTokenAuth`, `basic_auth`, `OAuth2ClientCredentials`, `build_auth` |
| `api/settings.py` | `ApiSettings` (the `modules.api` section) |
| `api/assertions.py` | `assert_status`, `assert_status_class`, `assert_latency`, `assert_header`, `assert_headers_present`, `assert_content_type`, `assert_redirect`, `assert_error_response`, `assert_json_keys`, `assert_json_subset`, `assert_field`, `assert_collection`, `assert_pagination`, `get_path` |
| `api/schema.py` | `schema_violations`, `assert_matches_schema`, `load_schema`, `check_schema` |
| `api/contract.py` | `OpenApiContract` (`from_file`, `from_url`, `validate_request`, `validate_response`, `validate_exchange`, `assert_response`), `find_breaking_changes` |
| `api/mocking.py` | `MockApi`, `Route`, `Reply` |
| `api/reporting.py` | `record_api_call`, `save_exchange`, `call_summary` |
| `api/diagnostics.py` | `Sanitizer` (URLs, headers and bodies, redacted) |
| `api/errors.py` | `ApiError`, `ApiTransportError`, `ApiTimeoutError`, `ResponseDecodeError`, `RedirectLoopError`, `UnsafeOperationError`, `AuthenticationError`, `ContractError` |
| `api/pytest_plugin.py` | Fixtures: `api_settings`, `api_client_factory`, `api_client`, `openapi_contract`, `mock_api` |

## Clients and lifecycle

```python
from complete_automation.api import ApiClient

with ApiClient("https://api.example.test/v1", timeout=10.0) as client:
    response = client.get("/users/{user_id}", path_params={"user_id": 42}, params={"expand": "roles"})
```

- **Connections:** one `ApiClient` owns one HTTPX connection pool. Close it with `close()` or a `with` block. Construction opens no connection.
- **URLs:** paths are relative to `base_url`, so `/users` becomes `/v1/users`. `{name}` placeholders are filled from `path_params`, URL-encoded, and checked: a missing, extra or empty value is an error. An absolute URL is accepted only on the same origin.
- **Request arguments:**
  - `params`: a dict or a list of pairs, for repeated keys.
  - `headers`, `cookies`.
  - Bodies: `json`, `data` (form), `files` (multipart), `content` (raw).
  - `timeout`: per request.
  - `auth=None`: send this request without credentials.
  - `follow_redirects`, `retry`.
  - `idempotency_key`, `idempotent`: make a POST or PATCH retryable.
- **Methods:** `get`, `post`, `put`, `patch`, `delete`, `head`, `options` and `request(method, ...)`.
- **HTTP errors are returned, not raised.** Only transport failures raise:
  - `ApiTransportError` (category `infrastructure`).
  - `ApiTimeoutError` (category `timeout`), with the original HTTPX error chained.
- **No implicit environment:** `trust_env` is off, so proxies and `.netrc` credentials are never picked up implicitly. Configure `proxy` explicitly.
- **Correlation ID:** each request carries `X-Correlation-ID`. It is the current test's correlation ID, or a new one.
- **Async:** `AsyncApiClient` has the same API with `await`, `async with` and `await aclose()`. Use it for concurrent calls with `asyncio.gather`. Use it inside your own coroutine, for example `asyncio.run(main())`; the fixtures provide synchronous clients.

`ApiResponse` wraps `httpx.Response`. `raw` is the original response, so nothing is hidden.

- **Body:** `status_code`, `headers`, `cookies`, `content`, `text`, `json()`, `content_type`, `is_json`. The body is never decoded implicitly. `json()` raises `ResponseDecodeError`, an `AssertionError`, for an empty or malformed body, with a redacted excerpt.
- **Timing:**
  - `latency_ms`: client-observed latency of the final attempt, from `perf_counter`.
  - `total_ms`: all attempts plus back-off.
  - `server_timing()`: server-reported durations from the `Server-Timing` header, kept separate from client latency.
  - `attempts`: one `AttemptRecord` per attempt.
  - `retries`, `redirects`.
- **Diagnostics:** `describe()` (for example `GET /users/{user_id} -> 404 (3.2 ms)`), `body_excerpt()`, and `to_record()`, a sanitized exchange for artifacts.

## Configuration

API settings live in `modules.api` of the shared configuration. They follow the same precedence as every other setting (see [shared_core.md](shared_core.md#configuration)).

```yaml
# config/environments/staging.yaml
modules:
  api:
    base_url: https://staging.example.test/v1
    timeouts: {connect: 5, read: 30, write: 30, pool: 5}   # finite by default
    follow_redirects: false
    max_redirects: 5
    default_headers: {Accept: application/json}
    latency_threshold_ms: 800
    retry: {enabled: true, max_attempts: 3, status_codes: [429, 502, 503, 504]}
    capture_exchanges: on_failure        # never | on_failure | always
    contract: {spec_path: data/api/contracts/users_api.openapi.yaml}
    auth: {type: bearer}                 # token from CAUTO_MODULES__API__AUTH__TOKEN
```

**TLS:** certificate verification is on. `verify_tls: false` is refused unless `allow_insecure_tls: true` is also set, and the client logs a warning. Use it only for local services. Prefer `ca_bundle` for private CAs.

**Protected environments:** `read_only` refuses POST, PUT, PATCH and DELETE with `UnsafeOperationError`. When unset, it is on in the `protected_environments` (`prod`, `production`). Mutating a protected environment requires `read_only: false` in that environment's configuration.

**Secrets:** keep credentials out of YAML; set them through `CAUTO_MODULES__API__...` variables or `.env`. Secret fields are `SecretStr`, and never appear in `repr`, logs or reports.

## Authentication

All strategies are `httpx.Auth` flows, so they work with both clients. A new strategy subclasses `httpx.Auth` with no change to the transport.

| Strategy | Configuration (`modules.api.auth`) | Notes |
| --- | --- | --- |
| `ApiKeyAuth(value, name="X-API-Key", location="header")` | `{type: api_key, name, location}` + `value` | Prefer headers; `location="query"` only when the API requires it (query values are masked in diagnostics) |
| `BearerTokenAuth(token)` or `(token_provider=fn, refresh=fn)` | `{type: bearer}` + `token` | `refresh` runs once on a 401, then the request is repeated |
| `basic_auth(username, password)` | `{type: basic, username}` + `password` | Encoded by HTTPX |
| `OAuth2ClientCredentials(token_url, client_id, client_secret, scopes=..., audience=..., client_auth="basic"\|"body")` | `{type: oauth2_client_credentials, ...}` | Token cached until `expires_in − leeway_seconds`; a 401 discards it and retries once; errors show endpoint, status and OAuth `error` code only |

Only the client-credentials grant is implemented. Other grants are out of scope until a project needs them.

## Validation

- **Assertions** (`api/assertions.py`):
  - Failure messages read like `POST /users -> 422 (4.1 ms): expected status 201, got 422`, followed by a redacted, truncated body excerpt.
  - `assert_field` checks one field: value, type (`bool` is not an `int`), nullability, numeric bounds, regex, allowed values, or a date format.
  - `assert_pagination` checks that the page, per-page, total and total-pages fields agree with the items returned.
- **JSON Schema** (`api/schema.py`):
  - The dialect comes from `$schema` (default 2020-12), and formats such as `email` and `date-time` are checked.
  - `assert_matches_schema` lists every violation by location (`body.data[0].id`).
- **OpenAPI** (`api/contract.py`):
  - Supports 3.0 and 3.1, and the document itself is validated first.
  - Checks path, query and header parameters, request bodies, documented status codes (exact, `2XX`, `default`), required response headers, content types and JSON bodies, with `$ref` resolved within the document.
  - `from_url` fetches only when called, with a timeout, and parses JSON or YAML safely.
  - `find_breaking_changes(old, new)` detects:
    - removed operations;
    - removed status codes;
    - new required parameters;
    - a request body that became required;
    - newly required top-level request properties;
    - top-level properties removed from JSON 2xx responses.
  - Not covered:
    - cookie parameters;
    - `deepObject` and delimited parameter styles;
    - non-JSON body schemas;
    - remote `$ref`;
    - security requirements;
    - nested-schema compatibility.

## Retries and timeouts

- **Timeouts:** connect, read, write and pool timeouts are finite by default. `timeout=` overrides them for one request.
- **Retries are off** unless `RetryConfig` (or `modules.api.retry.enabled`) is set. Then:
  - Retried: listed status codes (5xx, 408, 409, 425, 429; never 401, 403 or 422) and listed transport errors (connect, read timeout, pool timeout, protocol errors).
  - `Retry-After` is honoured (seconds or HTTP date), capped by `max_retry_after_seconds`. Otherwise the delay is exponential back-off with optional bounded jitter, from the core `RetryPolicy`.
  - **Method gate:** POST and PATCH are retried only with `idempotency_key=` (sent as `Idempotency-Key`) or `idempotent=True`. They cannot be added to `methods`.
- **On exhaustion:** retrying stops at `max_attempts`. The last response is returned, or `ApiTransportError` is raised with the attempt count. Every attempt is recorded in `response.attempts`.
- **No nested retries:** nothing else in the module retries, except the single token refresh after a 401. Do not wrap clients in another retry layer.

## Mocked tests versus integration tests

| Layer | Folder | Needs | Proves |
| --- | --- | --- | --- |
| Unit (mocked) | `tests/unit/api` | Nothing; `MockApi` / `httpx.MockTransport` | The client builds requests, handles responses, retries, redacts |
| Integration | `tests/integration/api` | A local service on 127.0.0.1 (`tests/support/api_server.py`) | Real HTTP: sockets, timeouts, redirects across origins, uploads |
| Examples | `tests/examples/api` | The same local service | How to write API tests |
| Live (opt-in) | `tests/live/api` | Configuration and credentials for a real API | That a real service behaves as expected |

A mocked test validates client behaviour; it does not show that a remote service implements the contract. Use contract validation against a running service for that.

**`MockApi`:**

- Routes use path templates.
- Replies can be a fixed response, a sequence (the last reply repeats), a transport error (`fail_with(httpx.ReadTimeout)`) or a handler function.
- Every request is recorded in `route.calls`.
- Unmatched requests get a 501 and can be checked with `assert_no_unmatched()`.

**Live tests** (`tests/live/api/test_reqres_live.py`, against reqres.in) skip unless `modules.api.base_url` is configured. Their expected payloads come from that service's public examples and have not been verified in this repository.

## Fixtures

`tests/unit/api/conftest.py`, `tests/integration/api/conftest.py`, `tests/examples/api/conftest.py` and `tests/live/api/conftest.py` load them with `from complete_automation.api.pytest_plugin import *`, and skip the folder when the extra is missing.

| Fixture | Scope | Notes |
| --- | --- | --- |
| `api_settings` | session | `framework_settings.section("api", ApiSettings)`, file paths resolved |
| `api_client_factory` | function | `factory(**overrides)` → `ApiClient.from_settings(...)`; all clients closed after the test, even on failure |
| `api_client` | function | Client for `modules.api.base_url`; skips the test if unset |
| `openapi_contract` | session | From `modules.api.contract`; skips if none configured |
| `mock_api` | function | A fresh `MockApi` |

Override any of these by defining a fixture with the same name in a closer `conftest.py`. None is autouse, and none touches the network while being set up.

## Reporting and artifacts

Clients from `api_client_factory` publish through the core plugin's `result_recorder`. Calls appear in the shared `TestResult`:

- **`metadata.api_calls`:** operation template, method, status, sanitized URL, latency, total time, retries, redirects and environment.
- **Metrics:** one `api.latency_ms` metric per call.
- **Artifacts:** `capture_exchanges: on_failure` (the default) stores the last 10 exchanges as redacted JSON under `artifacts/<run_id>/api/` and attaches their `ArtifactRef`s, when the test fails. `always` stores them on every test.

Failures are classified by the core:

| Failure | Status | Category |
| --- | --- | --- |
| Assertion helper or `assert` | `failed` | `assertion` |
| `ResponseDecodeError` | `failed` | `assertion` |
| Transport errors | `error` | `infrastructure` or `timeout` |
| `UnsafeOperationError` | `error` | `configuration` |

Run with `--cauto-report` to write the JSON report.

## Test data and parameterization

API test data uses the core loaders. Do not add another test-data framework.

```python
class CreateUserCase(BaseModel):
    case_id: str
    payload: dict
    expected_status: int
    expected_error_fields: list[str] = []

CASES = DataLoader(ROOT / "data").load_records("api/users/create_user_cases.yaml", CreateUserCase)

@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_create_user(client, case): ...
```

`data/api/users/create_user_cases.yaml` holds valid, boundary and invalid payloads. For per-environment data, keep one file per environment and choose it from `framework_settings.environment`. The xlsx loader from the core works the same way.

## Running API tests

```bash
uv run pytest -m api                                    # every API test (live ones skip unless configured)
uv run pytest tests/unit/api                            # mocked unit tests
uv run pytest tests/integration/api tests/examples/api  # local service
uv run pytest tests/unit/api/test_retry.py              # one file
uv run pytest "tests/unit/api/test_retry.py::test_parse_retry_after"   # one test
uv run pytest -m "api and not live" --cauto-report      # with a JSON report
CAUTO_ENVIRONMENT=reqres CAUTO_MODULES__API__AUTH__VALUE=<key> uv run pytest tests/live/api -m live
```

## Adding an endpoint or an authentication mechanism

**Endpoint:** call it with its path template, for example `client.get("/orders/{order_id}", path_params=...)`. Add it to the OpenAPI document if one exists, and add test data under `data/api/`. A thin helper function or class over `ApiClient` is welcome when a suite repeats the same calls. No base class is required.

**Authentication:** subclass `httpx.Auth` and implement `auth_flow`, or `sync_auth_flow` and `async_auth_flow` when the flow does I/O. Expose `sensitive_names` for any custom header or query parameter, and add a settings model with `build_auth` support if it should be configurable.

## Examples

### Runnable examples (offline, with `MockApi`)

```python
import asyncio
import httpx
from complete_automation.api import ApiClient, BearerTokenAuth, RetryConfig
from complete_automation.api.assertions import assert_error_response, assert_field, assert_status
from complete_automation.api.contract import OpenApiContract
from complete_automation.api.mocking import MockApi, Reply
from complete_automation.api.reporting import record_api_call, save_exchange
from complete_automation.api.schema import assert_matches_schema
from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.core.context import RunContext, utc_now
from complete_automation.core.pytest_plugin import ResultRecorder
from complete_automation.core.reporting import JsonReporter
from complete_automation.core.results import Status, TestResult
import tempfile
from pathlib import Path

USER = {"id": 1, "name": "Ada", "email": "ada@example.test", "role": "admin", "created_at": "2026-01-01T09:00:00Z"}
api = MockApi("https://api.example.test/v1")
api.route("GET", "/users/{user_id}").respond(200, json=USER)
create_route = api.route("POST", "/users").respond(201, json=USER, headers={"Location": "/v1/users/1"})
api.route("GET", "/secure").handle(lambda r: httpx.Response(200 if r.headers.get("Authorization") == "Bearer t-1" else 401, json={"error": "unauthorized"}))
api.route("GET", "/flaky").respond_sequence([Reply(503), Reply(200, json={"ok": True})])

# 1. Create a client (here on a mock transport; a real client just omits it)
client = api.client(sleep=lambda seconds: None)

# 2. A GET request
user = client.get("/users/{user_id}", path_params={"user_id": 1})
assert_status(user, 200)
assert_field(user, "email", pattern=r"[^@]+@[^@]+")

# 3. A JSON POST
created = client.post("/users", json={"name": "Ada", "email": "ada@example.test"})
assert_status(created, 201)

# 4. Bearer-token authentication
with api.client(auth=BearerTokenAuth("t-1")) as secured:
    assert_status(secured.get("/secure"), 200)

# 5. Validate a response schema
assert_matches_schema(user, {"type": "object", "required": ["id", "email"],
                             "properties": {"id": {"type": "integer"}, "email": {"type": "string", "format": "email"}}})

# 6. Validate against an OpenAPI contract
contract = OpenApiContract.from_file("data/api/contracts/users_api.openapi.yaml")
contract.assert_response(user, include_request=True)

# 7. An expected 4xx response
assert_error_response(client.get("/secure"), 401)

# 8. Retry behaviour (503, then 200)
flaky = client.get("/flaky", retry=RetryConfig(max_attempts=3))
assert flaky.status_code == 200 and flaky.retries == 1

# 9. A mocked test asserts on what the client sent
assert create_route.calls[0].headers["Content-Type"] == "application/json"
client.close()

# 10. Publish API results through the shared-core contracts
recorder = ResultRecorder()
record_api_call(recorder, user, environment="dev")
store = LocalArtifactStore(Path(tempfile.mkdtemp()) / "run-1")
recorder.attach(save_exchange(store, user))
run = RunContext.start("dev", run_id="run-1")
result = TestResult(run_id=run.run_id, test_id="docs::test_get_user", name="test_get_user", module="api",
                    status=Status.PASSED, started_at=run.started_at, finished_at=utc_now(),
                    metrics=tuple(recorder.metrics), artifacts=tuple(recorder.artifacts), metadata=recorder.metadata)
reporter = JsonReporter(store.root / "results.json")
reporter.start_run(run)
reporter.publish(result)
print(reporter.finalize(run.finish()).name, result.metadata["api_calls"][0]["status_code"], result.artifacts[0].path)
```

Expected output: `results.json 200 api/get_users_user_id_200.json`.

In a pytest suite, examples 1 and 10 come for free: `api_client_factory` builds and closes clients and records every call. See `tests/examples/api/test_users_api_examples.py`, which shows a successful request, a validation failure, an authentication failure, a schema mismatch, an expected server error and a contract failure.

## Limitations and troubleshooting

- **No API tests collected:** install the extra with `uv sync --extra api`.
- **`modules.api.base_url is not configured`:** `api_client` skips without it. Set it in the environment file or with `CAUTO_MODULES__API__BASE_URL`.
- **`UnsafeOperationError`:** either the environment is protected (read-only), or an absolute URL points at another origin. Create a separate client for that origin.
- **`ResponseDecodeError`:** the body is empty or is not JSON. Check `response.content_type` and `response.body_excerpt()`.
- **Retry concerns:** a retry did not happen because the method is not idempotent; use `idempotency_key`. To keep a request from repeating side effects, leave retries off; that is the default.
- **Performance:** latency assertions are functional checks on single requests. Throughput, saturation and concurrency limits need a dedicated performance-testing tool and strategy.
- **Redaction** is pattern-based, as described in the core. Do not put secrets in non-sensitive field names.
- **Proxy credentials:** `proxy` is passed to HTTPX as given. Keep credentials out of the URL when possible, and use a `SecretStr` setting from the environment otherwise.

## Phase 3 handoff: API and database integration points

The API module offers these for database testing without depending on it:

- `ApiResponse.json()` and `get_path(data, "data[0].id")` extract identifiers to look up in a database.
- `call_summary(response)` and `metadata.api_calls` record which API calls a test made, so a database check can be correlated through the same `TestResult`.
- The correlation ID sent as `X-Correlation-ID` is the test's `TestContext.correlation_id`. A database module can tag queries or audit rows with the same ID.
- `DataFactory` and `DataLoader` from the core should seed both API payloads and database fixtures, so the two stay consistent.
- `UnsafeOperationError` and `ApiSettings.is_read_only(environment)` show the protected-environment pattern a database module should mirror for writes.
