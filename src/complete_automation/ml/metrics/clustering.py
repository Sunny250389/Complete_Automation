"""Clustering metrics.

*Intrinsic* metrics (``silhouette``, ``davies_bouldin``, ``calinski_harabasz``) use only the
features and the cluster assignments. They need ``2 <= clusters <= samples - 1`` and are undefined
otherwise. Points labelled ``config.noise_label`` are left out of them. For more than 10 000
samples the silhouette is computed on a seeded random sample of 10 000 points (recorded in
``detail``) to bound memory.

*Extrinsic* metrics (``adjusted_rand_index``, ``normalized_mutual_info``) compare the assignments with
reference labels (the dataset's target column); they are unsupported without one. Cluster ids are
arbitrary names, and both metrics are invariant to renaming them.

Intrinsic scores favour compact, convex clusters and say little about whether the clustering is
meaningful for your task.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from complete_automation.ml.metrics.base import (
    MetricContext,
    MetricUndefined,
    MetricUnsupported,
    metric,
)
from complete_automation.ml.schemas import Direction, MetricStatus, MetricValue, TaskType

T = TaskType.CLUSTERING
HIGHER, LOWER = Direction.HIGHER_IS_BETTER, Direction.LOWER_IS_BETTER
SILHOUETTE_SAMPLE = 10_000


def _intrinsic_inputs(ctx: MetricContext, name: str) -> tuple[np.ndarray, np.ndarray]:
    if ctx.features is None:
        raise MetricUnsupported(f"{name} needs numeric features")
    labels = ctx.y_pred.astype(int)
    x = ctx.features
    if ctx.config.noise_label is not None:
        keep = labels != ctx.config.noise_label
        x, labels = x[keep], labels[keep]
    n, k = len(labels), len(np.unique(labels))
    if n < 3 or k < 2 or k > n - 1:
        raise MetricUndefined(
            f"{name} needs 2 <= clusters <= samples - 1; found {k} cluster(s) for {n} sample(s)"
        )
    return x, labels


@metric(T, "silhouette", HIGHER, "mean silhouette coefficient, in [-1, 1]")
def silhouette(ctx: MetricContext) -> float | MetricValue:
    x, labels = _intrinsic_inputs(ctx, "silhouette")
    if len(labels) > SILHOUETTE_SAMPLE:
        value = float(silhouette_score(x, labels, sample_size=SILHOUETTE_SAMPLE, random_state=ctx.config.seed))
        return MetricValue(name="silhouette", status=MetricStatus.COMPUTED, value=value, direction=HIGHER,
                           detail={"sampled": SILHOUETTE_SAMPLE, "seed": ctx.config.seed})
    return float(silhouette_score(x, labels))


@metric(T, "davies_bouldin", LOWER, "Davies-Bouldin index, 0 is best")
def davies_bouldin(ctx: MetricContext) -> float:
    x, labels = _intrinsic_inputs(ctx, "davies_bouldin")
    return float(davies_bouldin_score(x, labels))


@metric(T, "calinski_harabasz", HIGHER, "Calinski-Harabasz index (variance ratio)")
def calinski_harabasz(ctx: MetricContext) -> float:
    x, labels = _intrinsic_inputs(ctx, "calinski_harabasz")
    return float(calinski_harabasz_score(x, labels))


def _extrinsic(ctx: MetricContext, name: str) -> tuple[np.ndarray, np.ndarray]:
    reference = ctx.need_truth(name)
    if len(reference) < 2:
        raise MetricUndefined(f"{name} needs at least two samples")
    return reference, ctx.y_pred


@metric(T, "adjusted_rand_index", HIGHER, "agreement with reference labels, 1 is perfect, ~0 is chance")
def adjusted_rand_index(ctx: MetricContext) -> float:
    reference, labels = _extrinsic(ctx, "adjusted_rand_index")
    return float(adjusted_rand_score(reference, labels))


@metric(T, "normalized_mutual_info", HIGHER, "normalised mutual information with reference labels, in [0, 1]")
def normalized_mutual_info(ctx: MetricContext) -> float:
    reference, labels = _extrinsic(ctx, "normalized_mutual_info")
    return float(normalized_mutual_info_score(reference, labels))


INTRINSIC_METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz")
EXTRINSIC_METRICS = ("adjusted_rand_index", "normalized_mutual_info")
