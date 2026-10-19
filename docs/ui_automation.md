# UI automation

`complete_automation.ui` tests web applications in a real browser with [Playwright for Python](https://playwright.dev/python/) and pytest. It covers browser lifecycle, page objects, UI assertions, per-test failure diagnostics and fixtures.

Everything shared comes from [the shared core](shared_core.md): configuration, logging and redaction, results, reporting and artifact storage. The UI module adds no second version of any of them.

**Boundaries.**

- The module depends only on `complete_automation.core` and Playwright. It contains no API, database, ML or LLM code, and the core never imports it.
- Importing `complete_automation.ui` needs neither Playwright nor a browser. Nothing starts until a test uses a UI fixture.
- `tests/unit/ui/test_architecture.py` enforces these rules.
- There is no remote browser or grid support. Browsers run on the machine that runs pytest.

## Installation

```bash
uv sync --locked --extra ui            # Playwright's Python package
uv run playwright install chromium     # browser binaries; or: firefox, webkit, or all three
uv sync --locked --all-extras          # everything, as CI does
```

Without the `ui` extra the core still works and the UI test folders are left out of collection. If the extra is installed but a browser is not, the first UI test fails with `BrowserUnavailableError: chromium is not installed for Playwright; run: playwright install chromium`.

On Linux, `playwright install --with-deps <browser>` also installs the system libraries the browser needs. That form is used in CI and has not been run on this project's Windows development machine.

## Architecture

| Module | Public interface |
| --- | --- |
| `ui/settings.py` | `UiSettings` (the `modules.ui` section) |
| `ui/browser.py` | `BrowserSession` (Playwright runtime + browser), `launch_browser`, `launch_options`, `context_options` |
| `ui/pages/base_page.py` | `BasePage`: navigation, locators, actions and readers |
| `ui/pages/components.py` | `FormField`, `Notice` |
| `ui/assertions.py` | `assert_visible`, `assert_hidden`, `assert_text`, `assert_count`, `assert_enabled`, `assert_disabled`, `assert_attribute`, `assert_value`, `assert_checked`, `assert_selected_option`, `assert_validation_message`, `assert_url`, `assert_title` |
| `ui/diagnostics.py` | `UiDiagnostics`, `artifact_subdir` |
| `ui/pytest_plugin.py` | Fixtures `ui_settings`, `ui_session`, `ui_base_url`, `ui_context`, `ui_page` |
| `ui/errors.py` | `UiError`, `UiAssertionError`, `UiTimeoutError`, `BrowserUnavailableError` |

### Shared-core interfaces used

| Need | Core interface |
| --- | --- |
| Configuration | `FrameworkSettings.section("ui", UiSettings)`, `load_settings` (`modules.ui`, `CAUTO_MODULES__UI__*`) |
| Artifacts | `LocalArtifactStore` through the `artifact_store` fixture; `ArtifactRef`; `sanitize_name` |
| Results | `ResultRecorder` (`result_recorder`): `attach()` for artifacts, `add_metadata(ui=...)` |
| Per-test identity | `test_context`, `TestResult.test_id` (every artifact's metadata carries it) |
| Redaction | `Redactor` from `FrameworkSettings.redactor()`; `DEFAULT_REDACTOR` in assertion messages |
| Errors | `FrameworkError` / `ErrorCategory` / `classify_exception` |
| Logging | `get_logger("ui")` |

Only the pytest plugin, `ui/diagnostics.py` and `ui/browser.py` touch fixtures, the artifact store or Playwright's runtime; page objects and assertions work on any Playwright `Page` or `Locator`.

## Configuration

UI settings live in `modules.ui` of the shared configuration (`config/environments/<env>.yaml`, then `.env`, then `CAUTO_*` environment variables). Invalid values fail when the section is read, with the key in the message.

| Key | Default | Meaning |
| --- | --- | --- |
| `base_url` | none | Application root, `http(s)://host[:port]`, no credentials. Relative paths resolve against it. |
| `browser` | `chromium` | `chromium`, `firefox` or `webkit` |
| `headless` | `true` | `false` shows the browser window |
| `navigation_timeout_seconds` | `30` | Default for navigations |
| `action_timeout_seconds` | `10` | Default for clicks, fills and other actions |
| `expect_timeout_seconds` | `5` | Default wait of the UI assertions |
| `viewport` | `1280x720` | `{width, height}` |
| `locale`, `timezone_id` | browser default | For example `de-DE`, `Europe/Berlin` |
| `test_id_attribute` | `data-testid` | Attribute used by `by_test_id` |
| `launch_args` | none | Extra browser flags, each a single token such as `--mute-audio` |
| `slow_mo_ms` | `0` | Slow every Playwright call (debugging) |
| `screenshot` | `on_failure` | `never`, `on_failure` or `always` |
| `trace` | `off` | `off`, `on_failure` or `always` |
| `video` | `off` | `off`, `on_failure` or `always` |
| `max_screenshots_per_test` | `3` | Open pages captured per test |

`config/environments/ui_local.yaml` is a starting point for a local application:

```yaml
modules:
  ui:
    base_url: http://127.0.0.1:3000
    browser: chromium
    headless: true
```

### Choosing a browser, headless or headed

Set the value in YAML, or override it per run with an environment variable (the form below is the one used in this project's own verification):

```bash
CAUTO_MODULES__UI__BROWSER=firefox uv run pytest -m ui          # chromium, firefox or webkit
CAUTO_MODULES__UI__HEADLESS=false  uv run pytest -m ui          # headed
```

On Windows PowerShell: `$env:CAUTO_MODULES__UI__BROWSER = "firefox"; uv run pytest -m ui`.

### Secrets

No credential belongs in YAML, fixtures or page objects. Read them from environment variables inside the test (for example `os.environ["MYAPP_PASSWORD"]`) and fill the form with them. The framework never logs what a page object fills, and `base_url` is rejected if it contains credentials.

## Running UI tests

```bash
uv run pytest -m ui                          # every UI test (unit + browser)
uv run pytest tests/unit/ui                  # unit tests only: no browser is launched
uv run pytest tests/integration/ui           # framework self-tests: real browser, local site
uv run pytest tests/examples/ui              # runnable examples
uv run pytest tests/examples/ui --cauto-report   # also writes reports/<run_id>/results.json
```

### Framework self-tests versus tests of your application

`tests/integration/ui` and `tests/examples/ui` test the framework and demonstrate it. They run against a small static site in `tests/support/ui_site/`, served by the Python standard library on `127.0.0.1` and a free port (`tests/support/ui_site/server.py`, fixture `ui_site`). They need no network and no external site, and their `conftest.py` points `ui_settings` at that site.

To test your own application, write tests anywhere under `tests/` (outside those two folders), set `modules.ui.base_url` for your environment, and use the plugin's `ui_settings` unchanged:

```bash
uv run pytest tests/ui_app --cauto-env ui_local
```

The `ui_base_url` fixture skips a test when `modules.ui.base_url` is not configured.

## Writing a page object and a test

A page object holds a page's locators and user actions. A test holds the intent and the assertions.

```python
from playwright.sync_api import Page

from complete_automation.ui.assertions import assert_text, assert_url, assert_visible
from complete_automation.ui.pages import BasePage, FormField, Notice


class RegistrationPage(BasePage):
    path = "/form.html"

    def __init__(self, page: Page, base_url: str | None = None) -> None:
        super().__init__(page, base_url)
        self.name = FormField(page, "Full name", error_test_id="name-error")
        self.submit_button = self.by_role("button", "Create account")
        self.confirmation = Notice(page, "confirmation")

    def submit(self) -> None:
        self.click(self.submit_button)


def test_name_is_required(ui_page: Page, ui_base_url: str) -> None:
    form = RegistrationPage(ui_page, ui_base_url)
    form.open()

    form.submit()

    assert_text(form.name.error, "Full name is required")
```

The full pages are in `tests/support/ui_pages.py`; their tests are in `tests/examples/ui/test_ui_examples.py`.

Guidelines:

- Prefer `by_role`, `by_label`, `by_test_id`, then `by_text`. Avoid XPath and positional CSS selectors.
- Do not sleep. Actions wait for the element to be actionable; assertions poll until their timeout.
- `BasePage` actions turn a Playwright timeout into `UiTimeoutError` (category `timeout`) that names the action and the page. Other Playwright errors propagate unchanged.
- UI assertions raise `UiAssertionError`, an `AssertionError` (category `assertion`, so the result status is `failed`) that adds the page URL and title to Playwright's explanation and keeps the original as `__cause__`. Messages are redacted.
- Nothing retries a failed action. A flaky UI test is a defect to investigate, not to hide.
- `is_visible` and `count_of` are instantaneous reads; use the assertions to wait.

## Fixtures

| Fixture | Scope | Provides |
| --- | --- | --- |
| `ui_settings` | session | `UiSettings` from `modules.ui` |
| `ui_session` | module | `BrowserSession`: one Playwright runtime and one browser |
| `ui_base_url` | session | `modules.ui.base_url`; skips if unset |
| `ui_context` | function | A new, isolated `BrowserContext` with timeouts, viewport, tracing and video as configured |
| `ui_page` | function | A new `Page` in that context |

Each test gets its own context, so cookies, local storage and open pages never leak between tests. `ui_context` is closed after every test, passed or failed, even when the test or the diagnostics raise.

The plugin is registered once in `tests/conftest.py` when Playwright is installed:

```python
pytest_plugins = ["complete_automation.core.pytest_plugin", "complete_automation.ui.pytest_plugin"]
```

Register it in one place only; star-importing it into several `conftest.py` files creates one session fixture per folder and starts several Playwright runtimes.

## Failure diagnostics and artifacts

After each test, `ui_context` stores diagnostics through the shared `artifact_store` and attaches each `ArtifactRef` to that test's `TestResult` (`artifacts`):

| Artifact | When | Notes |
| --- | --- | --- |
| `screenshot-N.png` | `screenshot: on_failure` (default) or `always` | One per open page, up to `max_screenshots_per_test` |
| `console.json` | The test failed, or `screenshot: always` | Browser console messages and page errors, redacted, last 200 |
| `diagnostics.json` | Same | Test id, outcome, browser, version, viewport, page URLs, settings, redacted failure text |
| `trace.zip` | `trace: on_failure` or `always` | Playwright trace; open with `playwright show-trace` |
| `video-N.webm` | `video: on_failure` or `always` | Recorded for every test while enabled, kept per policy |

Files are stored under `<artifacts.root>/<run_id>/ui/<test name>-<8 hex of sha1(test id)>/`. The folder is unique per test and parameter set, and every artifact's `metadata` holds `test_id`, `kind` and `browser`. The result's `metadata.ui` records browser, version, headless mode, viewport, base URL and final page URLs for every UI test.

Safeguards:

- Collection never raises. If a screenshot, trace or video cannot be saved, the test keeps its original outcome and failure, the problem is logged and it is listed in `metadata.ui.diagnostic_errors`.
- Console text, URLs, failure text and error messages pass through the shared `Redactor`.
- **Screenshots, traces and videos are not redacted.** A screenshot shows what is on the screen; a trace contains page contents and network traffic, including request headers and cookies. Keep `trace` and `video` off unless you are investigating, and treat the artifact directory as sensitive when they are on.
- Traces and videos cost time and disk space, so both default to `off`.

## Known limitations

- Tests that call `asyncio.run` must not share a module with UI tests: while Playwright's sync runtime is alive the thread has a running event loop. The browser is therefore closed at the end of each test module.
- Only one Playwright sync runtime can exist per process. Tests of `BrowserSession` itself and of the failure diagnostics start their own runtime in a subprocess for this reason.
- Playwright's Python package, the browser binaries and, on Linux, system libraries must be installed separately from the Python lock file. Browser versions are bundled with the Playwright version in `uv.lock`.
- Parallel execution with `pytest-xdist` is not set up. Each worker would start its own browser, and artifact names are unique per test, but this has not been tested.
- No remote browsers, grids, mobile emulation, accessibility scans or visual comparison.
