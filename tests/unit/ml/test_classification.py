"""Classification metrics against values worked out by hand (not produced by the code under test)."""

import math

import pytest

from tests.unit.ml.helpers import evaluate_arrays

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]

# 4 positives, 6 negatives: TP=3, FN=1, FP=1, TN=5
Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
P_POS = [0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.2, 0.1, 0.1]


AVERAGED = tuple(f"{m}_{a}" for m in ("precision", "recall", "f1") for a in ("binary", "macro", "micro", "weighted"))


def binary(**kw):
    return evaluate_arrays("classification", Y_TRUE, Y_PRED, positive_label=1, **kw)


def test_binary_label_metrics() -> None:
    report = binary(metrics=("accuracy", "balanced_accuracy", "precision", "recall", "f1", *AVERAGED))
    assert report.value("accuracy") == pytest.approx(0.8)
    # positive class: P = 3/4, R = 3/4; negative class: P = R = 5/6; default averaging is macro
    assert report.value("precision_binary") == pytest.approx(0.75)
    assert report.value("recall_binary") == pytest.approx(0.75)
    assert report.value("f1_binary") == pytest.approx(0.75)
    macro = (0.75 + 5 / 6) / 2
    assert report.value("precision") == pytest.approx(macro)
    assert report.value("recall") == pytest.approx(macro)
    assert report.value("f1") == pytest.approx(macro)
    assert report.value("balanced_accuracy") == pytest.approx(macro)
    assert report.value("precision_weighted") == pytest.approx((0.75 * 4 + 5 / 6 * 6) / 10)
    assert report.value("recall_micro") == pytest.approx(0.8)  # micro P = R = F1 = accuracy


def test_confusion_matrix_rows_are_true_classes() -> None:
    detail = binary().metric("confusion_matrix").detail
    assert detail == {"labels": [0, 1], "matrix": [[5, 1], [1, 3]]}


def test_per_class_metrics_and_support() -> None:
    report = binary()
    assert report.value("recall[1]") == pytest.approx(0.75)
    assert report.value("precision[0]") == pytest.approx(5 / 6)
    assert report.value("support[0]") == 6 and report.value("support[1]") == 4


def test_probability_metrics() -> None:
    report = binary(proba=[[1 - p, p] for p in P_POS], classes=[0, 1])
    # 23 of the 24 (positive, negative) pairs are ranked correctly
    assert report.value("roc_auc") == pytest.approx(23 / 24)
    # positives at ranks 1, 2, 3 and 5 of 10: (1 + 1 + 1 + 4/5) / 4
    assert report.value("average_precision") == pytest.approx(0.95)
    true_class_probability = [p if y == 1 else 1 - p for y, p in zip(Y_TRUE, P_POS)]
    expected = -sum(math.log(p) for p in true_class_probability) / 10
    assert report.value("log_loss") == pytest.approx(expected)


def test_decision_scores_support_auc_but_not_log_loss() -> None:
    report = binary(scores=[2 * p - 1 for p in P_POS], classes=[0, 1], metrics=("roc_auc", "log_loss"))
    assert report.value("roc_auc") == pytest.approx(23 / 24)
    assert report.metric("log_loss").status.value == "unsupported"


def test_probability_metrics_are_unsupported_from_hard_labels_alone() -> None:
    report = binary(metrics=("roc_auc", "average_precision", "log_loss"))
    for name in ("roc_auc", "average_precision", "log_loss"):
        assert report.metric(name).status.value == "unsupported", name
        assert report.metric(name).value is None


def test_multiclass_metrics() -> None:
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "c", "c"]
    report = evaluate_arrays("classification", y_true, y_pred, metrics=("accuracy", "precision", "recall", "f1", "recall_weighted", "per_class"))
    assert report.value("accuracy") == pytest.approx(4 / 6)
    assert report.value("precision") == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert report.value("recall") == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert report.value("f1") == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert report.value("recall_weighted") == pytest.approx((2 / 3 * 3 + 0.5 * 2 + 1) / 6)
    assert report.value("f1[a]") == pytest.approx(0.8)  # non-integer labels work


def test_multiclass_roc_auc_is_one_vs_rest_macro() -> None:
    y_true = ["a", "b", "c", "a", "b", "c"]
    proba = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 2
    report = evaluate_arrays("classification", y_true, ["a", "b", "c"] * 2, proba=proba, classes=["a", "b", "c"])
    assert report.value("roc_auc") == pytest.approx(1.0)
    assert report.value("average_precision") == pytest.approx(1.0)


def test_probability_columns_follow_the_model_class_order() -> None:
    # columns are given in the order b, a; the evaluator must realign them to the sorted classes
    y_true = ["a", "a", "b", "b"]
    proba_b_a = [[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]]
    report = evaluate_arrays("classification", y_true, ["a", "a", "b", "b"], proba=proba_b_a,
                             classes=["b", "a"], positive_label="b")
    assert report.value("roc_auc") == pytest.approx(1.0)
    assert report.value("log_loss") == pytest.approx(-sum(map(math.log, [0.9, 0.7, 0.8, 0.6])) / 4)


def test_binary_averaging_needs_an_explicit_positive_label() -> None:
    report = evaluate_arrays("classification", Y_TRUE, Y_PRED, metrics=("f1_binary", "roc_auc"),
                             proba=[[1 - p, p] for p in P_POS], classes=[0, 1])
    assert report.metric("f1_binary").status.value == "unsupported"
    assert "positive_label" in (report.metric("roc_auc").reason or "")


def test_a_class_that_is_never_predicted_follows_the_zero_division_policy() -> None:
    y_true, y_pred = [0, 0, 1, 1], [0, 0, 0, 0]
    as_zero = evaluate_arrays("classification", y_true, y_pred, metrics=("precision", "recall"))
    assert as_zero.value("precision") == pytest.approx((0.5 + 0.0) / 2)  # class 1: 0/0 counted as 0
    assert as_zero.value("recall") == pytest.approx(0.5)
    assert any(f.code == "zero_division" and "class 1" in f.message for f in as_zero.findings)

    strict = evaluate_arrays("classification", y_true, y_pred, metrics=("precision", "recall"),
                             zero_division="undefined")
    assert strict.metric("precision").status.value == "undefined"
    assert "1" in (strict.metric("precision").reason or "")
    assert strict.value("recall") == pytest.approx(0.5)


def test_one_class_targets_make_auc_undefined_but_not_accuracy() -> None:
    report = evaluate_arrays(
        "classification", [1, 1, 1], [1, 1, 0], labels=(0, 1), positive_label=1,
        proba=[[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]], classes=[0, 1], metrics=("accuracy", "roc_auc"))
    assert report.value("accuracy") == pytest.approx(2 / 3)
    auc = report.metric("roc_auc")
    assert auc.status.value == "undefined" and "one class" in auc.reason
    assert any(f.code == "single_class_target" for f in report.findings)


def test_an_unused_declared_class_is_reported_not_hidden() -> None:
    report = evaluate_arrays("classification", ["a", "a", "b"], ["a", "b", "b"], labels=("a", "b", "c"),
                             metrics=("recall",), zero_division="undefined")
    assert report.metric("recall").status.value == "undefined"  # class c has no samples
    assert "'c'" in report.metric("recall").reason


def test_imbalance_is_a_warning_and_never_fails_the_report() -> None:
    report = evaluate_arrays("classification", [0] * 18 + [1, 1], [0] * 20, metrics=("accuracy",))
    assert report.passed
    codes = {f.code: f.severity.value for f in report.findings}
    assert codes["class_imbalance"] == "warning"
    assert report.value("accuracy") == pytest.approx(0.9)  # high accuracy, yet the minority class is never found
    assert evaluate_arrays("classification", [0] * 18 + [1, 1], [0] * 20, metrics=("recall_binary",),
                           positive_label=1).value("recall_binary") == 0.0
