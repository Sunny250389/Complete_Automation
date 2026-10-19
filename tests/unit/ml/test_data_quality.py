"""Dataset loading and validation, prediction validation, and invalid-row handling."""

import json

import numpy as np
import pandas as pd
import pytest

from complete_automation.core.exceptions import TestDataError
from complete_automation.ml import (
    ColumnSpec,
    Dataset,
    DatasetError,
    DatasetLoader,
    DatasetSchema,
    EvaluationConfig,
    Evaluator,
    PrecomputedAdapter,
)
from tests.unit.ml.helpers import evaluate_arrays

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]


def check(frame, *, schema=None, task="classification", target="label", **config):
    dataset = Dataset.from_frame(frame, "d", features=["x"], target=target, schema=schema)
    adapter = PrecomputedAdapter(list(frame[target]) if target in frame else [0] * len(frame))
    return Evaluator(EvaluationConfig(task=task, metrics=("accuracy",) if task == "classification" else None,
                                      **config)).evaluate(dataset, adapter)


def codes(report):
    return {f.code for f in report.errors}


def good(**extra):
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1], **extra})


def test_a_valid_dataset_has_no_errors() -> None:
    assert check(good()).passed


def test_empty_dataset() -> None:
    report = check(pd.DataFrame({"x": [], "label": []}))
    assert codes(report) == {"empty_dataset"} and not report.passed and report.n_samples == 0


def test_missing_required_columns_are_named() -> None:
    dataset = Dataset.from_frame(pd.DataFrame({"x": [1.0, 2.0]}), "d", features=["x", "z"], target="label")
    report = Evaluator(EvaluationConfig(task="classification")).evaluate(dataset, PrecomputedAdapter([0, 1]))
    assert codes(report) == {"missing_columns"}
    assert set(report.errors[0].details["columns"]) == {"z", "label"}


def test_a_null_target_is_a_blocking_error_with_a_count() -> None:
    report = check(pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": [0, None, 1]}))
    assert codes(report) == {"null_values"} and report.errors[0].details == {"column": "label", "count": 1}


def test_null_features_warn_unless_the_schema_forbids_them() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "label": [0, 1, 0, 1]})
    assert check(frame).passed and any(f.code == "null_values" for f in check(frame).warnings)
    schema = DatasetSchema(columns={"x": ColumnSpec(dtype="numeric", nullable=False)})
    assert codes(check(frame, schema=schema)) == {"null_values"}


def test_unexpected_dtype() -> None:
    frame = good().assign(x=["a", "b", "c", "d"])
    report = check(frame, schema=DatasetSchema(columns={"x": ColumnSpec(dtype="numeric")}))
    assert codes(report) == {"unexpected_dtype"} and report.errors[0].details["expected"] == "numeric"


def test_non_integer_values_fail_an_integer_column() -> None:
    schema = DatasetSchema(columns={"x": ColumnSpec(dtype="integer")})
    assert codes(check(good().assign(x=[1.5, 2.0, 3.0, 4.0]), schema=schema)) == {"unexpected_dtype"}
    assert check(good(), schema=schema).passed  # integral floats such as 1.0 are integers


def test_non_finite_values_are_blocking() -> None:
    report = check(pd.DataFrame({"x": [1.0, np.inf, 3.0], "label": [0, 1, 0]}))
    assert codes(report) == {"non_finite_values"} and report.errors[0].details["count"] == 1


def test_unexpected_categories_and_out_of_range_values() -> None:
    schema = DatasetSchema(columns={"colour": ColumnSpec(dtype="string", allowed_values=("red", "blue")),
                                    "x": ColumnSpec(dtype="numeric", min=0.0, max=3.5)})
    report = check(good(colour=["red", "blue", "green", "red"]), schema=schema)
    assert codes(report) == {"unexpected_values", "out_of_range"}


def test_extra_columns_can_be_forbidden() -> None:
    schema = DatasetSchema(columns={"x": ColumnSpec()}, forbid_extra_columns=True)
    assert codes(check(good(secret=[1, 2, 3, 4]), schema=schema)) == {"unexpected_columns"}


def test_duplicates_warn_by_default_and_can_block() -> None:
    frame = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0], "label": [0, 0, 1, 1]})
    report = check(frame)
    assert report.passed and [f.details["count"] for f in report.warnings if f.code == "duplicate_records"] == [1]
    assert codes(check(frame, schema=DatasetSchema(duplicates="error"))) == {"duplicate_records"}
    assert check(frame, schema=DatasetSchema(duplicates="off")).warnings == ()


def test_unique_key_duplicates() -> None:
    frame = pd.DataFrame({"id": [1, 1, 2], "x": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    schema = DatasetSchema(unique_key=("id",), duplicates="error")
    assert codes(check(frame, schema=schema)) == {"duplicate_records"}


def test_invalid_target_labels_against_declared_labels() -> None:
    report = check(good().assign(label=[0, 1, 2, 1]), labels=(0, 1))
    assert "invalid_target_labels" in codes(report)


def test_class_imbalance_is_reported_with_counts_and_does_not_fail() -> None:
    frame = pd.DataFrame({"x": np.arange(12.0), "label": [0] * 10 + [1] * 2})
    report = check(frame)
    imbalance = next(f for f in report.warnings if f.code == "class_imbalance")
    assert imbalance.details["counts"] == {"0": 10, "1": 2} and report.passed


def test_regression_target_must_be_numeric() -> None:
    report = check(good().assign(label=["a", "b", "c", "d"]), task="regression")
    assert "non_numeric_target" in codes(report)


def test_findings_never_contain_feature_values() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "name": ["patient-7-secret", None, "b", "c"],
                          "label": [0, 1, 0, 1]})
    schema = DatasetSchema(columns={"name": ColumnSpec(dtype="string", allowed_values=("b",))})
    dataset = Dataset.from_frame(frame, "d", features=["x", "name"], target="label", schema=schema)
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",))).evaluate(
        dataset, PrecomputedAdapter([0, 1, 0, 1]))
    assert report.errors  # the unexpected value was found ...
    assert "patient-7-secret" not in report.model_dump_json()  # ... and counted, not quoted


def test_class_names_do_appear_in_reports_because_per_class_metrics_need_them() -> None:
    assert "class-x" in evaluate_arrays("classification", ["class-x", "class-y"], ["class-x", "class-y"]).model_dump_json()


# --- invalid rows -----------------------------------------------------------------------------


def test_invalid_rows_can_be_excluded_with_a_report_of_why() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.inf, 5.0], "label": [0, 1, None, 1, 0]})
    dataset = Dataset.from_frame(frame, "d", features=["x"], target="label")
    adapter = PrecomputedAdapter(pd.Series([0, 1, 1, 1, 0]))
    report = Evaluator(EvaluationConfig(task="classification", metrics=("accuracy",),
                                        invalid_rows="exclude")).evaluate(dataset, adapter)
    assert report.n_samples == 2 and report.n_excluded == 3
    excluded = next(f for f in report.findings if f.code == "rows_excluded")
    assert excluded.details["count"] == 3
    assert excluded.details["reasons"] == {"missing_target": 1, "missing_feature": 1, "non_finite_value": 1}
    assert report.value("accuracy") == 1.0  # rows 0 and 4 survive, and their predictions stay aligned


def test_nothing_is_excluded_when_the_policy_is_error() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "label": [0, None]})
    assert check(frame).n_excluded == 0


# --- prediction validation -------------------------------------------------------------------


def test_prediction_and_label_length_mismatch() -> None:
    dataset = Dataset.from_frame(pd.DataFrame({"x": [1.0, 2.0, 3.0], "t": [0, 1, 0]}), "d", features=["x"], target="t")
    report = Evaluator(EvaluationConfig(task="classification")).evaluate(dataset, PrecomputedAdapter([0, 1]))
    assert codes(report) == {"prediction_length_mismatch"} and report.metrics == ()


def test_unexpected_predicted_labels() -> None:
    report = evaluate_arrays("classification", [0, 1, 0], [0, 1, 7], labels=(0, 1))
    assert codes(report) == {"unexpected_predicted_labels"} and report.errors[0].details["count"] == 1


@pytest.mark.parametrize(
    ("proba", "code"),
    [
        ([[0.5, 0.5, 0.0]] * 3, "invalid_probability_shape"),
        ([[0.5, 0.5]] * 2, "invalid_probability_shape"),
        ([[0.6, 0.6], [0.5, 0.5], [0.5, 0.5]], "probabilities_not_normalized"),
        ([[1.2, -0.2], [0.5, 0.5], [0.5, 0.5]], "probability_out_of_range"),
        ([[np.nan, 1.0], [0.5, 0.5], [0.5, 0.5]], "non_finite_probabilities"),
    ],
    ids=["extra-column", "missing-row", "sum-not-one", "out-of-range", "nan"],
)
def test_invalid_probabilities_block_the_evaluation(proba, code) -> None:
    report = evaluate_arrays("classification", [0, 1, 0], [0, 1, 0], proba=proba, classes=[0, 1])
    assert code in codes(report) and report.metrics == ()


def test_probabilities_must_cover_the_evaluated_classes() -> None:
    report = evaluate_arrays("classification", [0, 1, 2], [0, 1, 2], proba=[[0.5, 0.5]] * 3, classes=[0, 1])
    assert "invalid_probability_shape" in codes(report)


def test_invalid_score_shape() -> None:
    report = evaluate_arrays("classification", [0, 1, 0], [0, 1, 0], scores=np.ones((3, 5)), classes=[0, 1])
    assert codes(report) == {"invalid_score_shape"}


def test_a_two_dimensional_prediction_is_an_unexpected_shape() -> None:
    dataset = Dataset.from_frame(pd.DataFrame({"x": [1.0, 2.0], "t": [1.0, 2.0]}), "d", features=["x"], target="t")
    from complete_automation.ml import FunctionAdapter

    adapter = FunctionAdapter(lambda X: np.ones((len(X), 3)), model_id="wide")
    report = Evaluator(EvaluationConfig(task="regression")).evaluate(dataset, adapter)
    assert codes(report) == {"unexpected_prediction_shape"}


# --- loader -----------------------------------------------------------------------------------


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_loading_types_and_metadata(tmp_path) -> None:
    write(tmp_path / "d.csv", "x,colour,label\n1.5,red,0\n2.5,,1\n3.5,blue,0\n")
    schema = DatasetSchema(columns={"colour": ColumnSpec(dtype="string", nullable=True)})
    dataset = DatasetLoader(tmp_path).load("d.csv", features=["x", "colour"], target="label", schema=schema,
                                           dataset_id="demo", version="v1")
    assert dataset.metadata.dataset_id == "demo" and dataset.metadata.version == "v1"
    assert dataset.metadata.n_rows == 3 and dataset.metadata.source == "d.csv"
    assert dataset.frame["x"].dtype == "float64" and dataset.frame["label"].dtype == "int64"
    assert dataset.frame["colour"].isna().sum() == 1  # an empty cell is missing, not an empty string
    assert dataset.metadata.fingerprint == DatasetLoader(tmp_path).load(
        "d.csv", features=["x", "colour"], target="label", schema=schema).metadata.fingerprint


def test_json_records_loading(tmp_path) -> None:
    write(tmp_path / "d.json", json.dumps([{"x": 1, "label": "a"}, {"x": 2, "label": "b"}]))
    dataset = DatasetLoader(tmp_path).load("d.json", features=["x"], target="label")
    assert list(dataset.frame["label"]) == ["a", "b"]


def test_the_fingerprint_changes_when_data_changes() -> None:
    a = Dataset.from_frame(good(), "d", features=["x"], target="label")
    b = Dataset.from_frame(good().assign(x=[1.0, 2.0, 3.0, 4.5]), "d", features=["x"], target="label")
    assert a.metadata.fingerprint != b.metadata.fingerprint


def test_loader_refuses_bad_inputs(tmp_path) -> None:
    write(tmp_path / "d.parquet", "x")
    write(tmp_path / "d.json", '{"not": "records"}')
    write(tmp_path / "big.csv", "x,label\n" + "1,0\n" * 10)
    loader = DatasetLoader(tmp_path, max_rows=5)
    with pytest.raises(DatasetError, match="unsupported"):
        loader.load("d.parquet", features=["x"])
    with pytest.raises(DatasetError, match="list of records"):
        loader.load("d.json", features=["x"])
    with pytest.raises(DatasetError, match="max_rows"):
        loader.load("big.csv", features=["x"])
    with pytest.raises(DatasetError, match="max_file_bytes"):
        DatasetLoader(tmp_path, max_file_bytes=5).load("big.csv", features=["x"])
    with pytest.raises(TestDataError):
        loader.load("../outside.csv", features=["x"])
    with pytest.raises(TestDataError):
        loader.load("missing.csv", features=["x"])


def test_a_ragged_csv_is_rejected_by_the_shared_loader(tmp_path) -> None:
    write(tmp_path / "d.csv", "x,label\n1,0\n2\n")
    with pytest.raises(TestDataError):
        DatasetLoader(tmp_path).load("d.csv", features=["x"])
