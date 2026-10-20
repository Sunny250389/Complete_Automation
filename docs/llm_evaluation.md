# LLM evaluation

`complete_automation.llm_evaluation` evaluates a language model by **calling it**, capturing the response it actually generated, and checking that response against expectations: deterministic checks, an optional separate judge model, quality gates and baselines. Results go through [the shared core](shared_core.md): configuration (`modules.llm_evaluation`), `TestResult`/`ResultRecorder`, `ArtifactStore`, the data loaders, the `Redactor` and the error categories.

Ollama is the supported provider. The model under test is always invoked; a judge is optional and is a different component that only ever sees an already captured response.

**Boundaries.** The module imports only `complete_automation.core`, `httpx`, `jsonschema` and pydantic. It contacts only the `base_url` you configure, downloads and pulls nothing, never executes model output, and never falls back to a mock. The core and other modules never import it. `tests/unit/llm_evaluation/test_architecture.py` enforces this. The older DeepEval prototype (`llm/`, `metrics/`, `tests/llm/`) is untouched.

## Three kinds of test, kept apart

| Kind | Where | Needs Ollama | What it proves |
| --- | --- | --- | --- |
| **Mocked / offline** | `tests/unit/llm_evaluation`, `tests/integration/llm_evaluation/test_pytest_integration.py` | No | Client parsing and error handling against a scripted HTTP transport; deterministic metrics; judge parsing and orchestration (the judge's JSON is supplied by the test); gates, baselines, reports, redaction, fixtures. **It does not show that any real model works.** |
| **Real Ollama** | `tests/integration/llm_evaluation/test_ollama_live.py` (marker `llm_live`) | Yes | A real, installed model was called, its actual response was captured and evaluated, and the result was reported. Opt-in. |
| **Real judge** | `test_real_judge_scores_a_real_model_answer` (marker `llm_live`) | Yes | A second (judge) model returned valid structured scores for a real answer. The scores are estimates, not truth. |

## Installation and prerequisites

```bash
uv sync --locked --extra llm          # httpx, jsonschema (also included in --all-extras)
```

Install [Ollama](https://ollama.com), start it (`ollama serve`, or the desktop app), and install a model yourself; nothing here pulls models:

```bash
ollama list                           # names of installed models, for example qwen3:4b
ollama pull <model>                   # only if you want another one
curl http://localhost:11434/api/tags  # is the server reachable, and what does it report?
```

## Configuration

Settings live under `modules.llm_evaluation` (`config/environments/<env>.yaml`) or `CAUTO_MODULES__LLM_EVALUATION__*` variables:

| Key | Default | Meaning |
| --- | --- | --- |
| `base_url` | `http://localhost:11434` | Ollama endpoint |
| `model` | none | The model under test. **Never defaulted**: set it to a name from `ollama list` |
| `judge_model`, `judge_base_url` | none, `base_url` | Optional judge, configured independently |
| `timeout_seconds` | `120` | Per request |
| `live` | `false` | Opt in to tests that call a real model |
| `require_model` | `false` | With `live`: an unreachable endpoint or missing model **fails** instead of skipping |
| `persist_raw` | `true` | `false`: reports and artifacts keep scores, outcomes, hashes, lengths, latency and token counts, but not prompts, responses, references, context or text that quotes them |
| `publish_artifacts`, `data_root`, `baseline_dir` | `true`, `data/llm_evaluation`, `data/llm_evaluation/baselines` | |

`config/environments/ollama_local.yaml` turns `live` on with a 300 s timeout (CPU-only machines are slow) and leaves the model to you.

## Running it

```bash
uv run pytest tests/unit/llm_evaluation                    # offline, no Ollama
uv run pytest -m llm_evaluation                            # offline + mocked integration; live tests skip
# the real thing:
export CAUTO_MODULES__LLM_EVALUATION__MODEL=qwen3:4b       # any installed model
export CAUTO_MODULES__LLM_EVALUATION__JUDGE_MODEL=qwen3:4b # optional; may be a different model
uv run pytest tests/integration/llm_evaluation/test_ollama_live.py --cauto-env ollama_local -s -v
uv run pytest -m llm_live --cauto-env ollama_local --cauto-report   # also writes reports/<run_id>/results.json
```

PowerShell: `$env:CAUTO_MODULES__LLM_EVALUATION__MODEL = "qwen3:4b"`. Without opt-in the live tests **skip with the reason**; with `live` on and the model missing they skip, or fail when `require_model` is true. They never report success without a real response.

## Architecture

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Immutable contracts: `ModelConfig`, `GenerationConfig`, `PromptTemplate`, `TestCase`, `CheckSpec`, `CriterionSpec`, `ModelRequest`, `ModelResponse`, `MetricOutcome`, `JudgeResult`, `EvaluationRecord`, `SuiteReport`, `ThresholdRule`, `BaselineRule` |
| `clients/base.py`, `clients/ollama.py` | `ModelClient` protocol; `OllamaClient` (`/api/chat`, `/api/generate`, `/api/tags`, streaming, availability) |
| `datasets.py` | Loading (JSONL, JSON, YAML, CSV, xlsx) and validation |
| `evaluators/deterministic.py` | Checks that need no judge |
| `evaluators/rubrics.py`, `evaluators/judge.py` | Versioned rubrics; `JudgeEvaluator` |
| `runner.py` | `EvaluationRunner`: request → call → capture → evaluate → gate |
| `gates.py`, `baseline.py` | Thresholds, aggregates, baseline files and comparison |
| `reporting.py` | `publish_report` (artifacts), `record_result` (`TestResult` metrics and metadata) |
| `pytest_plugin.py`, `settings.py` | Fixtures, `modules.llm_evaluation` settings |

**Raw data and verdicts are separate.** A `ModelResponse` is captured once and never edited. `EvaluationRunner.evaluate_response(case, response)` evaluates it without calling the model, so the same response can be judged by other checks later. Metadata a provider did not expose (token counts, `done_reason`, model digest) is `None`, never estimated.

**Failures are classified.** A record's status is `passed`; `failed` (a check, judged criterion or threshold was not met); `model_error` (the call failed, timed out, or a stream ended early: nothing complete to evaluate); or `evaluator_error` (a response exists but an evaluator or the judge could not reach a verdict, or a gated metric was unavailable). A demonstrated failure outranks an evaluator problem, and both are listed. `SuiteReport.assert_passed()` raises `ModelInvocationError`, `EvaluatorError` or `QualityGateError` accordingly; the first two never look like a pass. Skips are decided by pytest (`llm_live_model`).

### Adding a provider

Implement `ModelClient` (`clients/base.py`): `generate(request) -> ModelResponse` (normalised; raise a `ModelInvocationError` subclass when nothing was produced; no hidden retries) and `identify(config) -> str | None`. Pass it as `client=` to `EvaluationRunner` / `llm_evaluate`, or as `judge_client=` for the judge. No metric code changes. An OpenAI-compatible client is not included.

## Datasets

One test case per line in a `.jsonl` file (also `.json`, `.yaml`, `.csv`, `.xlsx`; in CSV/xlsx put structured fields such as `checks` in JSON text). Loading is confined to `data_root`. Validation reports every problem by record and field, never the content, before anything runs. Fields:

| Field | |
| --- | --- |
| `case_id`, `prompt` | required; ids are unique |
| `system_prompt` | optional |
| `reference` | reference answer, for `exact_match` and the `correctness` criterion |
| `expected_schema` | JSON Schema, default for `json_schema` |
| `context` | retrieved passages (a string or a list), for RAG cases |
| `checks`, `criteria`, `thresholds` | what to evaluate (at least one) |
| `tags`, `metadata` | free |

```json
{"case_id": "rag-launch-year", "prompt": "In which year was the Zephyr library first released?",
 "context": ["The Zephyr library was first released in 2014 by the Northwind team."],
 "reference": "Zephyr was first released in 2014.",
 "checks": [{"type": "contains", "value": "2014"}, {"type": "context_overlap", "minimum": 0.3}],
 "criteria": [{"name": "groundedness", "min_score": 4}, {"name": "correctness", "min_score": 4}]}
```

`data/llm_evaluation/examples/` holds `ollama_smoke.jsonl` and `rag_cases.jsonl`: small, offline, no private content. `uv run pytest tests/unit/llm_evaluation/test_datasets.py` validates them. The target model sees only `prompt` (and `system_prompt`); **the reference is never sent to it**. `context` reaches it only through a template that uses `{context}` (for example `RAG_TEMPLATE`); otherwise it is only for evaluation.

## Deterministic checks

Each returns passed, failed, or **error** (the check could not be evaluated, for example a missing reference or an invalid regex or schema). Errors never count as passes. Name a check with `name` when a case uses one type twice.

| `type` | `value` / options | Notes |
| --- | --- | --- |
| `not_empty` | | empty or whitespace-only fails |
| `exact_match` | text (default: the reference); `case_sensitive`, `normalize_whitespace` | only these transformations: strip and collapse whitespace, `casefold`. No punctuation or Unicode normalisation |
| `contains`, `not_contains` | string or list | required / forbidden text; lists cover required sections or fields |
| `regex` | pattern | `re.search` |
| `json_valid` | | accepts one surrounding Markdown code fence |
| `json_schema` | schema (default: `expected_schema`) | JSON Schema via `jsonschema`; invalid JSON is reported as *not valid JSON*, distinct from a schema violation |
| `json_keys` | required keys; `allowed_keys` | reports missing and unexpected keys |
| `length` | `minimum`, `maximum`, `unit` (`chars`/`words`) | |
| `refusal` | `true` (must refuse) / `false`; `patterns` | **keyword heuristic**: misses differently worded refusals, flags quoted refusal phrases; not a safety classifier |
| `completed` | | request completed and `done_reason` is not `length` (truncation). If Ollama reports no reason, it says truncation cannot be ruled out |
| `context_overlap` | `minimum` (required) | share of the answer's content words found in the context: a **lexical proxy**, not a groundedness verdict |

Latency and token limits are **thresholds on measurements**, not checks: every record has `latency_seconds`, `response_chars` and, if Ollama reported it, `output_tokens` (never inferred from characters). Add `{"metric": "latency_seconds", "operator": "le", "value": 30}` to a case's `thresholds`; a limit on an unmeasured metric is an evaluator error, not a pass. Language detection and semantic similarity are not implemented.

## Judge (optional)

```python
JudgeConfig(model=ModelConfig(model="<judge model>", generation=GenerationConfig(temperature=0.0)),
            scale_min=1, scale_max=5, repeats=1)
```

Built-in rubrics (versioned, in `evaluators/rubrics.py`): `correctness` (agreement with the reference only), `relevance`, `completeness`, `clarity`, `instruction_following`, `groundedness` (supported by the context, **not** true in the real world: a response can be grounded in a source that is wrong), `context_relevance` (judges the context, not the answer). Or give a custom `rubric` and `rubric_version`. Each criterion declares what the judge may see: groundedness gets the context but not the reference; correctness gets the reference but not the context.

* The judge must return one JSON object (`criterion`, `score`, `verdict`, `rationale`, `evidence`); the framework asks Ollama for that schema and validates the result. Bad JSON, a missing field, a wrong criterion, an out-of-scale score or an `uncertain` verdict is outcome **error**.
* `min_score` on the criterion decides pass/fail; without it the score is recorded only.
* `repeats > 1` calls the judge several times and reports every score, min, max and standard deviation, not just the mean. Any failed repeat makes the judgement an error. A handful of runs is not a statistical estimate.
* The response, reference, context and question are untrusted. They are wrapped in markers carrying a random per-call token, and the judge is told to treat them as data. **This reduces prompt-injection risk but does not remove it.**
* Judge scores are estimates from a language model. They are not ground truth, a security certification or proof of factual correctness. They are reported as `kind: "judge"`, apart from deterministic results, with the judge model and rubric version.

## Quality gates and baselines

Aggregates (see `gates.py`): per metric `<name>.pass_rate`, `.mean`, `.errors` (`<name>` is a check name or `judge.<criterion>`), plus `case_pass_rate`, `model_error_rate`, `evaluator_error_rate`, `latency_seconds.mean/max`, `output_tokens.mean`. Errors and un-evaluated cases count against pass rates and add nothing to means.

```python
EvaluationConfig(
    model=...,
    suite_thresholds=(
        ThresholdRule(metric="json_schema.pass_rate", operator="ge", value=1.0),
        ThresholdRule(metric="judge.groundedness.mean", operator="ge", value=4.0),
        ThresholdRule(metric="latency_seconds.max", operator="le", value=30.0),
    ),
    baseline_rules=(BaselineRule(metric="case_pass_rate", max_drop=0.05),),
)
```

Operators: `ge`, `gt`, `le`, `lt`, `eq` (with `tolerance`). **A rule on a metric that was not measured fails** (`metric_unavailable`) unless `required=False`. Thresholds are per suite or case; nothing is built in.

Baselines hold aggregate numbers only. Save one from a good run, adopt it by copying the file into `baseline_dir`, and compare later:

```python
Baseline.from_report(report, "qwen3-smoke-v1").save(artifact_store)       # then copy into data/llm_evaluation/baselines/
baseline = Baseline.load(settings.baseline_dir, "qwen3-smoke-v1.json")    # or the llm_load_baseline fixture
report = apply_baseline(report, baseline)                                 # or llm_evaluate(..., baseline=baseline)
```

A different output text, or a changed value, is reported but is **not** a failure; only a `BaselineRule` (an allowed worsening, in the metric's direction) can fail. A baseline for another dataset is refused; a changed dataset or model is a warning.

## Prompts and generation settings

`PromptTemplate(template_id, version, system, user)` with `{prompt}` and `{context}` placeholders; `RAG_TEMPLATE` is provided. The template label, the full `ModelConfig` (including generation settings; unset values are not sent), the judge configuration and rubric versions are stored in every report, so two runs with different prompt versions or settings can be compared from their reports. Nothing changes a system prompt or sampling setting between cases. To compare prompt versions, run the same dataset and model with two templates and compare the reports or baselines.

Reasoning models: leave `think` unset. In our check with `qwen3:4b`, `think=False` put the reasoning into the answer text, while the default returned it in the separate `thinking` field. Give such models enough `max_tokens`; a truncated response fails the `completed` check.

## Reports and artifacts

`llm_evaluate` (fixture) or `publish_report` writes, through the shared `ArtifactStore`, under `llm_evaluation/`: `evaluation-<id>.json` (run id, dataset id and fingerprint, target and judge model and provider, endpoint, template, generation configuration, per-case request/response/reference/context, every metric and judge result, gate and baseline results, latency and token metadata, errors, warnings), `responses-<id>.jsonl`, `cases-<id>.csv`, `metrics-<id>.csv`. The `TestResult` gets `llm.<metric>` metrics with gate verdicts and an `llm_evaluation` metadata block; add `--cauto-report` for `reports/<run_id>/results.json`.

Everything passes through the shared `Redactor` (credentials in keys, URLs, `Authorization` headers, `password=...` pairs) before storage. That is pattern-based and best effort. For sensitive prompts set `persist_raw: false`.

## Runnable example

```python
from pathlib import Path

from complete_automation.core.artifacts import LocalArtifactStore
from complete_automation.llm_evaluation import (
    CheckSpec, EvaluationConfig, EvaluationRunner, GenerationConfig, ModelConfig,
    OllamaClient, TestCase, ThresholdRule, publish_report,
)

config = EvaluationConfig(
    model=ModelConfig(model="qwen3:4b", timeout_seconds=300,           # an installed model
                      generation=GenerationConfig(temperature=0.0, seed=1, max_tokens=400)),
    suite_thresholds=(ThresholdRule(metric="case_pass_rate", operator="ge", value=1.0),),
)
case = TestCase(                                                        # 1. the test prompt
    case_id="capital-france",
    prompt="What is the capital of France? Answer in one short sentence.",
    checks=(CheckSpec(type="completed"), CheckSpec(type="contains", value="paris", case_sensitive=False)),
)
with OllamaClient() as client:                                          # 2. invoke the model
    report = EvaluationRunner(client, config, run_id="example").run([case], dataset_id="example")
record = report.records[0]                                              # 3. the captured, real response
print(record.response.model, record.response.model_identifier, repr(record.response.text))
print(record.status.value, report.aggregates["case_pass_rate"])         # 4./5. evaluation and gate
store = LocalArtifactStore(Path("artifacts") / "example")
report, refs = publish_report(report, store)                            # 6. JSON/JSONL/CSV artifacts
print([r.path for r in refs])
report.assert_passed()                                                  # raises if the gate failed
```

Under pytest the same flow is `llm_evaluate(cases_or_dataset, config, assert_passed=True)`, which also records the result and artifacts.

## Known limitations

* Ollama only. No OpenAI-compatible client, no semantic-similarity or embedding metric, no language detection.
* Judge scores are model opinions; small judges are noisy. Prompt delimiting reduces but does not prevent injection.
* `refusal` and `context_overlap` are heuristics, as described above.
* Sampling with `temperature=0` and a `seed` is *more* repeatable, not guaranteed identical across hardware or Ollama versions.
* Requests are never retried. Latency figures are wall-clock for one request, not benchmarks; there is no load testing.
* Streaming: an interrupted stream returns `partial` text and is a model error unless `evaluate_partial` is set.
* The model digest comes from a separate `/api/tags` call and is `None` if that fails.
* Redaction is pattern-based; it cannot recognise every secret.
* `uv.lock` is generated by `uv lock` (it was regenerated in Phase 8, with no version changes, after an earlier hand edit for this extra).
