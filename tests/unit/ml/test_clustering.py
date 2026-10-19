"""Clustering metrics against hand-computed values on a four-point example."""

import pandas as pd
import pytest

from tests.unit.ml.helpers import evaluate_arrays

pytestmark = [pytest.mark.ml, pytest.mark.cauto_module("ml")]

X = pd.DataFrame({"x": [0.0, 1.0, 10.0, 11.0]})
GOOD = [0, 0, 1, 1]


def run(labels=GOOD, reference=None, features=X, **config):
    return evaluate_arrays("clustering", reference, labels, features=features, **config)


def test_intrinsic_metrics() -> None:
    report = run()
    # silhouette: points 0 and 11 give 1 - 1/10.5, points 1 and 10 give 1 - 1/9.5
    assert report.value("silhouette") == pytest.approx(0.5 * (9.5 / 10.5 + 8.5 / 9.5))
    # each cluster has scatter 0.5 and the centroids are 10 apart: (0.5 + 0.5) / 10
    assert report.value("davies_bouldin") == pytest.approx(0.1)
    # between-cluster SS 100, within-cluster SS 1: (100 / (2 - 1)) / (1 / (4 - 2))
    assert report.value("calinski_harabasz") == pytest.approx(200.0)


def test_extrinsic_metrics_ignore_cluster_names() -> None:
    report = run(labels=[5, 5, 3, 3], reference=[0, 0, 1, 1])
    assert report.value("adjusted_rand_index") == pytest.approx(1.0)
    assert report.value("normalized_mutual_info") == pytest.approx(1.0)


def test_unrelated_assignments_score_below_chance() -> None:
    report = run(labels=[0, 1, 0, 1], reference=[0, 0, 1, 1])
    assert report.value("adjusted_rand_index") == pytest.approx(-0.5)
    assert report.value("normalized_mutual_info") == pytest.approx(0.0)


def test_extrinsic_metrics_are_unsupported_without_reference_labels() -> None:
    report = run(metrics=("silhouette", "adjusted_rand_index"))
    assert report.metric("silhouette").status.value == "computed"
    ari = report.metric("adjusted_rand_index")
    assert ari.status.value == "unsupported" and "reference" in ari.reason


def test_default_metrics_include_extrinsic_ones_only_with_a_reference() -> None:
    assert {m.name for m in run().metrics} == {"silhouette", "davies_bouldin", "calinski_harabasz"}
    assert "adjusted_rand_index" in {m.name for m in run(reference=[0, 0, 1, 1]).metrics}


@pytest.mark.parametrize(
    ("labels", "why"),
    [([0, 0, 0, 0], "found 1 cluster"), ([0, 1, 2, 3], "found 4 cluster")],
    ids=["one-cluster", "every-sample-its-own-cluster"],
)
def test_invalid_cluster_counts_are_undefined_with_an_explanation(labels, why) -> None:
    report = run(labels=labels)
    for name in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        item = report.metric(name)
        assert item.status.value == "undefined" and item.value is None
        assert why in item.reason


def test_a_single_sample_is_undefined() -> None:
    report = evaluate_arrays("clustering", None, [0], features=pd.DataFrame({"x": [1.0]}))
    assert report.metric("silhouette").status.value == "undefined"


def test_noise_points_are_left_out_of_intrinsic_metrics() -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 10.0, 11.0, 500.0]})
    report = run(labels=[0, 0, 1, 1, -1], features=features, noise_label=-1, metrics=("silhouette",))
    assert report.value("silhouette") == pytest.approx(0.5 * (9.5 / 10.5 + 8.5 / 9.5))


def test_non_integer_cluster_ids_are_rejected() -> None:
    report = run(labels=[0.5, 0.5, 1.0, 1.0])
    assert [f.code for f in report.errors] == ["invalid_cluster_assignments"]
