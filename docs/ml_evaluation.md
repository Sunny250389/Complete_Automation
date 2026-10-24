# ML model evaluation

`complete_automation.ml` evaluates machine-learning models: it takes a dataset, ground-truth labels where they exist, and a model's predictions, validates them, computes task-specific metrics, applies quality gates and reports the outcome through the shared core. It is an **evaluation** module. It never trains, tunes, saves or serves a model.

Everything shared comes from [the shared core](shared_core.md): configuration (`modules.ml`), `TestResult`/`ResultRecorder`, `ArtifactStore`, the data loaders, the `Redactor` and the error categories. The ML module adds no second version of any of them.

**Boundaries.** The module imports only `complete_automation.core` plus numpy, pandas and scikit-learn. It has no HTTP, database, browser or LLM code, and the core and the other modules never import it. LLM evaluation is a separate module and is out of scope here. `tests/unit/ml/test_architecture.py` enforces this, and also that the module contains no `fit` calls, no pickle/joblib loading and no `eval`/`exec`.

## Installation and commands

```bash
uv sync --locked --extra ml            # numpy, pandas, scikit-learn
uv sync --locked --all-extras          # everything, as CI does

uv run pytest tests/unit/ml                       # framework unit tests (hand-computed expected values)
uv run pytest tests/integration/ml                # real scikit-learn models, artifacts, pytest plugin
uv run pytest tests/examples/ml                   # example model-quality tests
uv run pytest -m ml                               # all of the above
uv run pytest -m ml --cauto-report                # also write reports/<run-id>/results.json
```

Without the extra, the core still works and the ML test folders are left out of collection.

## Architecture

| Module | Responsibility |
| --- | --- |
| `ml/schemas.py` | Typed contracts: `EvaluationConfig`, `ThresholdRule`, `MetricValue`, `GateOutcome`, `ValidationFinding`, `DatasetSchema`, `EvaluationReport` and the rest. Immutable pydantic models, JSON-serialisable |
| `ml/data.py` | `Dataset` (features, target, metadata, fingerprint) and `DatasetLoader` (CSV/JSON, confined to a data root, size limits) |
| `ml/validation.py` | Dataset checks, invalid-row detection, prediction and probability checks |
| `ml/models.py` | `ModelAdapter` and `PrecomputedAdapter`, `EstimatorAdapter`, `FunctionAdapter`; `Predictions` |
| `ml/metrics/` | Metric registry plus the classification, regression and clustering metrics |
| `ml/thresholds.py` | Quality gates (`evaluate_gates`, `parse_thresholds`) |
| `ml/baseline.py` | `Baseline` files and `compare_to_baseline` |
| `ml/robustness.py` | Perturbations and repeatability helpers (opt-in) |
| `ml/evaluator.py` | `Evaluator`: validate, predict, measure, gate, compare |
| `ml/reporting.py` | `publish_report` (artifacts) and `record_result` (`TestResult` metrics and metadata) |
| `ml/pytest_plugin.py` | Fixtures: `ml_settings`, `ml_dataset_loader`, `ml_load_baseline`, `ml_evaluate` |
| `ml/fixtures.py` | Seeded synthetic datasets for tests and examples |

Evaluation results are separate from model output: `Predictions` (arrays) is what a model returns; `EvaluationReport` holds metrics, findings, gate outcomes and metadata, never raw rows or predictions.

### Error rule

* Invalid **data or predictions** never raise from `Evaluator.evaluate`. They become *blocking findings* (`severity: error`) on the report, no metrics are computed, and `report.passed` is false. `report.assert_passed()` raises `ValidationFailedError`.
* Invalid **configuration** (unknown metric, unsupported task, contradictory options) raises `MLConfigurationError` or a pydantic `ValidationError` before anything runs.
* A model that raises while predicting raises `ModelInvocationError`.
* A metric that cannot be computed is **never** replaced by a number: it is `undefined` (the data make it undefined) or `unsupported` (the task or the model outputs cannot provide it), with a reason.
* Warnings (class imbalance, nulls in features, zero-division, undefined metrics, duplicates by default) never fail a report by themselves. `on_undefined="fail"` makes undefined metrics fail it.

| Cause | Error class | Shared-core category / `TestResult` status |
| --- | --- | --- |
| Threshold or baseline rule violated | `QualityGateError` | `threshold` / failed |
| Gated metric undefined or unsupported | `MetricUnavailableError` | `test_data` / error |
| Blocking validation finding | `ValidationFailedError` | `test_data` / error |
| Configuration problem | `MLConfigurationError` | `configuration` / error |
| Model raised | `ModelInvocationError` | `unknown` / error |

## Supported tasks and metrics

| Task | Metrics (default set in bold) | Direction |
| --- | --- | --- |
| `classification` | **`accuracy`**, **`balanced_accuracy`**, **`precision`**, **`recall`**, **`f1`**, **`confusion_matrix`**, **`per_class`** (`precision[c]`, `recall[c]`, `f1[c]`, `support[c]`); with scores or probabilities: **`roc_auc`**, **`average_precision`**; with probabilities: **`log_loss`**. Explicit averaging: `precision_binary`, `recall_macro`, `f1_weighted`, `f1_micro`, ... | higher is better, except `log_loss` |
| `regression` | **`mae`**, **`mse`**, **`rmse`**, **`r2`**, **`mape`**, **`median_ae`** | lower is better, except `r2` |
| `clustering` | **`silhouette`**, **`davies_bouldin`**, **`calinski_harabasz`**; with reference labels: **`adjusted_rand_index`**, **`normalized_mutual_info`** | `davies_bouldin` lower, others higher |

Set `EvaluationConfig.metrics` to choose; unknown names for the task raise `MLConfigurationError`. Multilabel classification is not supported (deferred).

### Metric behaviour and limitations

* **Classification, imbalanced data.** `accuracy` rewards predicting the majority class: 18 negatives and 2 positives predicted all-negative score 0.90 with recall 0 for the positive class. Read it with `balanced_accuracy` and the per-class metrics. Imbalance is reported as a warning with class counts (`imbalance_warning_ratio`, default 5), never as a failure.
* **Averaging.** `precision`/`recall`/`f1` use `config.averaging` (default `macro`). `binary` needs `config.positive_label`; the framework never guesses the positive class. Binary `roc_auc` and `average_precision` also need it.
* **Zero division.** `zero_division="zero"` (default) counts a 0/0 precision/recall/F1 as 0 and adds a `zero_division` warning naming the classes. `"undefined"` makes the affected metric undefined instead.
* **Probabilities.** `roc_auc`, `average_precision` and `log_loss` need `predict_proba` (or `decision_function` for the first two) and are `unsupported` from hard labels. Probabilities must have shape `(rows, classes)`, be finite, within [0, 1] and sum to 1 (`probability_atol`). Columns are realigned from the model's `classes_` order. Multiclass `roc_auc`/`average_precision` are one-vs-rest macro averages and are undefined if any class is absent from the targets.
* **`r2`** compares with predicting the mean: undefined for constant targets, negative for models worse than the mean, not comparable across datasets.
* **`mape`** is a fraction (0.10 = 10 %). Any target with `abs(y) <= mape_epsilon` makes it undefined, unless `mape_zero_policy="exclude"` drops those rows from this metric only (the count is in `detail`). It explodes for targets near zero.
* **Clustering.** Intrinsic metrics need `2 <= clusters <= samples - 1`; otherwise they are undefined with the cluster count in the reason. `noise_label` leaves noise points (for example `-1`) out of intrinsic metrics. Silhouette uses a seeded 10 000-sample subset for larger inputs. ARI and NMI are invariant to cluster names and unsupported without reference labels.

## Datasets and validation

`DatasetLoader(root)` reads **CSV** and **JSON** (a list of records) with the shared-core loaders, confined to `root`. Parquet is not supported (it would add a binary dependency); convert to CSV. Files are read fully into memory, so `max_file_bytes` (50 MB) and `max_rows` (100 000) refuse larger inputs rather than streaming them. Empty CSV cells are missing values; undeclared columns are converted to numbers when every value parses, declared `string` columns stay text.

```python
schema = DatasetSchema(columns={
    "age": ColumnSpec(dtype="integer", min=0, max=120),
    "plan": ColumnSpec(dtype="string", allowed_values=("free", "pro")),
    "churned": ColumnSpec(dtype="integer", allowed_values=(0, 1)),
}, unique_key=("customer_id",), duplicates="error")
dataset = loader.load("churn.csv", features=["age", "plan"], target="churned", schema=schema, version="2025-01")
```

| Finding code | Severity | Meaning |
| --- | --- | --- |
| `empty_dataset`, `missing_columns`, `missing_target` | error | nothing to evaluate / required column absent |
| `unexpected_dtype`, `non_numeric_target`, `non_numeric_features` | error | declared or required type violated |
| `null_values` | error for the target and declared non-nullable columns; warning for undeclared features | |
| `non_finite_values` | error | infinities in numeric columns |
| `unexpected_values`, `out_of_range`, `unexpected_columns`, `invalid_target_labels` | error | value contracts |
| `duplicate_records` | warning (default), `error`/`off` through `schema.duplicates` | whole-row or `unique_key` duplicates |
| `class_distribution`, `class_imbalance`, `single_class_target` | info / warning | reported, never blocking |
| `prediction_length_mismatch`, `unexpected_prediction_shape`, `missing_predictions`, `non_finite_predictions`, `non_numeric_predictions`, `unexpected_predicted_labels`, `invalid_cluster_assignments` | error | prediction checks (`labels` declares the valid classes) |
| `invalid_probability_shape`, `probability_classes_mismatch`, `non_finite_probabilities`, `probability_out_of_range`, `probabilities_not_normalized`, `invalid_score_shape`, `non_finite_scores` | error | probability and score checks |

Findings carry counts and column names, never cell values. With `invalid_rows="exclude"`, rows with a missing target, missing features or infinite numbers are dropped, predictions of a `pandas.Series` stay aligned by index, and a `rows_excluded` warning records the count per reason. Nothing is dropped silently.

## Connecting a model

```python
from complete_automation.ml import EstimatorAdapter, FunctionAdapter, PrecomputedAdapter

EstimatorAdapter(fitted_sklearn_model, model_id="churn-logreg", version="3")   # predict / predict_proba / decision_function
FunctionAdapter(lambda X: client.predict(X), model_id="served", predict_proba=..., classes=[0, 1])
PrecomputedAdapter(labels, probabilities=proba, classes=[0, 1], model_id="batch-run-42")
PrecomputedAdapter.from_frame(frame, "prediction", probability_columns={0: "p_neg", 1: "p_pos"})
```

Methods a model lacks are simply not used. The evaluator passes the model a copy of the features, so a model cannot alter the dataset. **Extension point:** subclass `ModelAdapter` and implement `predict` (optionally `predict_proba`, `decision_function`, `classes`). **Serialised models are not loaded** by this module, because unpickling an untrusted file executes arbitrary code. Load a model you trust in your own code and wrap it.

## Quality gates

```python
config = EvaluationConfig(
    task="classification", positive_label=1,
    thresholds=(ThresholdRule.at_least("f1_binary", 0.90),
                ThresholdRule.at_least("recall[1]", 0.95),
                ThresholdRule.at_most("log_loss", 0.30)),
)
# or from configuration data:   thresholds=parse_thresholds({"f1_binary": {"ge": 0.9}, "rmse": {"le": 5.0}})
```

Operators: `ge`, `gt`, `le`, `lt`, `eq` (with `tolerance`). A rule is explicit; there are no default thresholds. A missing, undefined or unsupported metric **fails** its rule (`metric_unavailable`, distinct from `threshold_failed`) unless the rule has `required=False`. Every outcome records metric, observed value, operator, threshold and reason. A rule whose operator opposes the metric's direction (for example `ge` on `rmse`) yields a warning. `report.gates.passed` is the aggregate; `report.passed` additionally requires no blocking findings and passing baseline, robustness, repeatability and custom checks.

## Baseline comparison

```python
baseline = ml_load_baseline("churn_logreg.json")        # fixture; or Baseline.load(dir, name)
report = ml_evaluate(dataset, model, config, baseline=baseline,
                     baseline_rules=[BaselineRule(metric="recall[1]", max_drop=0.02),
                                     BaselineRule(metric="log_loss", max_relative_drop=0.10)])
```

A baseline is a small JSON file of metric values (no data). Create one from a report with `Baseline.from_report(report).save(artifact_store)`, review it, and copy it into `modules.ml.baseline_dir` (default `data/ml/baselines`). Every shared metric gets absolute and relative differences (`relative_difference` is `None` when the baseline is 0 or undefined). A change alone never fails; only a `BaselineRule` does, and it respects the metric's direction (an *increase* in `log_loss` is a worsening). If either value is undefined the ruled metric is `unavailable`, which fails a required rule. A `baseline_dataset_differs` warning appears when the dataset fingerprint differs.

## Robustness, repeatability and subgroups (opt-in)

```python
PerturbationSpec(name="noise", kind="gaussian_noise", features=("age",), magnitude=0.05, seed=7,
                 gate_metric="f1_binary", max_metric_drop=0.05, max_prediction_change_rate=0.10)
RepeatabilityConfig(runs=3)                                      # deterministic: exact equality
RepeatabilityConfig(runs=5, stochastic=True, atol=0.02)          # stochastic: stated tolerance
SubgroupConfig(column="region", min_group_size=30)
```

* **Perturbations:** `gaussian_noise` (std = `magnitude` x feature std), `relative_scale` (±`magnitude`), `set_missing` (only with `allow_missing_inputs=True`, declaring that the model accepts missing values), `substitute_category` (explicit `mapping`). They work on copies, are seeded, and are recorded (kind, features, magnitude, seed, cells changed). Reports show the prediction change and each metric's degradation; limits are optional and a prediction change is not assumed to be a defect. This is a sensitivity check, not an adversarial-robustness claim. Not available for precomputed predictions.
* **Subgroups:** descriptive metrics for an explicitly named column, with sample counts; groups below `min_group_size` are marked `insufficient` and not scored; more than `max_groups` distinct values is an error. Sensitive attributes are never inferred and the result is not a fairness verdict.
* **Extension point for fairness/explainability:** implement `EvaluationCheck` (`name`, `run(CheckContext) -> CheckResult`) and pass `checks=[...]`. No fairness or explainability implementation is included.

## Running under pytest

```python
def test_churn_model_quality(ml_dataset_loader, ml_evaluate):
    dataset = ml_dataset_loader.load("churn.csv", features=["age", "plan"], target="churned")
    model = EstimatorAdapter(load_my_trusted_model(), model_id="churn", version="3")
    config = EvaluationConfig(task="classification", positive_label=1,
                              thresholds=(ThresholdRule.at_least("recall[1]", 0.95),))
    ml_evaluate(dataset, model, config, assert_passed=True)      # a genuine pytest failure with diagnostics
```

Load the fixtures in a `conftest.py` (`from complete_automation.ml.pytest_plugin import *`); the folders under `tests/` already do. `ml_evaluate` returns the report without raising unless `assert_passed=True`, so a test can also assert on `report.value("f1")`, `report.gates`, `report.baseline` and so on. A failing gate reads, for example:

```
QualityGateError: ML evaluation failed:
  - threshold failed: recall[1] observed 0.91, required >= 0.95
```

Mark ML tests `@pytest.mark.ml` (the marker is registered). Thresholds are per test; none is mandatory.

## Reports and artifacts

`ml_evaluate` records on the test's `TestResult`: a `ml.<metric>` `MetricResult` for each metric (with `threshold` and `passed` when a rule gates it; an undefined metric is recorded as the text `undefined`, never as a number) and `metadata.ml_evaluation` (ids, sample counts, verdict, failure reasons). Through the shared `artifact_store` it writes, under `ml/`: `evaluation-<id>.json` (the full report: metadata, model and dataset ids and versions, task, sample counts, findings, metrics, gates, baseline, robustness, subgroups, reproducibility, artifact references), `metrics-<id>.csv`, and for classification `confusion-matrix-<id>.csv`. Disable with `modules.ml.publish_artifacts: false`.

Reports hold metrics, counts, column names and **class names** (per-class metrics need them), but no raw rows or predictions. The JSON passes through the shared `Redactor`. `reproducibility` records the config hash, dataset fingerprint, seeds and numpy/pandas/scikit-learn/Python versions; `evaluation_id` is a hash of config, dataset fingerprint and model id/version, so identical evaluations get identical ids and metrics. Confusion matrices are CSV; no plotting dependency is used.

Configuration (`config/environments/<env>.yaml`):

```yaml
modules:
  ml:
    data_root: data/ml
    baseline_dir: data/ml/baselines
    max_rows: 100000
    publish_artifacts: true
```

## Complete example

`data/ml/examples/binary_predictions.csv` holds 12 rows with a label, a prediction and class probabilities. It is exercised in `tests/examples/ml/test_ml_examples.py`. Standalone:

```python
from pathlib import Path
from complete_automation.ml import (DatasetLoader, EvaluationConfig, Evaluator, PrecomputedAdapter, ThresholdRule)

dataset = DatasetLoader(Path("data/ml")).load(
    "examples/binary_predictions.csv", features=["x1", "x2"], target="label", dataset_id="binary-predictions")
model = PrecomputedAdapter.from_frame(dataset.frame, "prediction", probability_columns={0: "p_neg", 1: "p_pos"},
                                      model_id="example-model")
config = EvaluationConfig(task="classification", positive_label=1,
                          thresholds=(ThresholdRule.at_least("accuracy", 0.80), ThresholdRule.at_least("recall[1]", 0.90)))
report = Evaluator(config).evaluate(dataset, model)
print(report.passed, report.value("accuracy"), report.failure_reasons)
# False 0.8333333333333334 ('threshold failed: recall[1] observed 0.8333333333, required >= 0.9',)
```

## Adding a metric or an evaluator

```python
from complete_automation.ml.metrics import MetricContext, MetricUndefined, metric
from complete_automation.ml.schemas import Direction, TaskType

@metric(TaskType.REGRESSION, "max_error", Direction.LOWER_IS_BETTER, "largest absolute error")
def max_error(ctx: MetricContext) -> float:
    y = ctx.need_truth("max_error").astype(float)
    return float(abs(y - ctx.y_pred.astype(float)).max())
```

Register the function with `@metric(task, name, direction)`. Return a float (or a `MetricValue` / list of them for structured results), and raise `MetricUndefined` or `MetricUnsupported` with a reason when it cannot be computed. Import the module before evaluating so the registration runs, then use the name in `EvaluationConfig.metrics` and in threshold rules. Add tests with independently computed expected values. A new task type needs a `TaskType` member, its validation in `validation.py` and its default metrics in `Evaluator._metric_names`.

## Known limitations

* Multilabel classification, Parquet input, streaming of large datasets and image plots are not implemented.
* Evaluation is single-process and in-memory; it is intended for test-sized datasets.
* Subgroup analysis, fairness and explainability: only the descriptive subgroup report and the `EvaluationCheck` extension point exist.
* Robustness and repeatability need a live model; precomputed predictions cannot be re-predicted.
* Baselines are plain JSON files managed by hand; there is no model registry or drift monitoring.
* Dataset fingerprints depend on the pandas version, so a baseline recorded with a different pandas may show a `baseline_dataset_differs` warning even for identical data.
