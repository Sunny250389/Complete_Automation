"""Small deterministic synthetic datasets for tests, examples and documentation.

Everything is generated from a seeded ``numpy.random.Generator``: no files, no network, no
downloads. The datasets are well separated on purpose, so lightweight models fit them quickly and
stably; they exercise the *evaluation framework*, not any real model's quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from complete_automation.ml.data import Dataset


def binary_classification(n: int = 200, seed: int = 0, positive_share: float = 0.5) -> Dataset:
    """Two Gaussian blobs; labels ``0``/``1``; features ``x1``, ``x2``."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_share).astype(int)
    centre = np.where(y[:, None] == 1, [1.2, 1.2], [-1.2, -1.2])
    x = centre + rng.normal(0.0, 1.0, (n, 2))
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": y})
    return Dataset.from_frame(frame, "synthetic-binary", features=["x1", "x2"], target="label", version=f"seed{seed}")


def multiclass_classification(n: int = 240, seed: int = 0) -> Dataset:
    """Three Gaussian blobs; string labels ``a``/``b``/``c``."""
    rng = np.random.default_rng(seed)
    names = np.array(["a", "b", "c"])
    index = rng.integers(0, 3, n)
    centres = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
    x = centres[index] + rng.normal(0.0, 1.0, (n, 2))
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": names[index]})
    return Dataset.from_frame(frame, "synthetic-multiclass", features=["x1", "x2"], target="label", version=f"seed{seed}")


def regression(n: int = 150, seed: int = 0, noise: float = 0.5) -> Dataset:
    """``y = 3*x1 - 2*x2 + 5 + noise``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, (n, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0 + rng.normal(0.0, noise, n)
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y})
    return Dataset.from_frame(frame, "synthetic-regression", features=["x1", "x2"], target="y", version=f"seed{seed}")


def clustering(n_per_cluster: int = 40, seed: int = 0) -> Dataset:
    """Three well-separated blobs; ``reference`` holds the generating cluster (0, 1, 2)."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-6.0, -6.0], [6.0, -6.0], [0.0, 7.0]])
    x = np.vstack([c + rng.normal(0.0, 0.8, (n_per_cluster, 2)) for c in centres])
    reference = np.repeat([0, 1, 2], n_per_cluster)
    frame = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "reference": reference})
    return Dataset.from_frame(frame, "synthetic-clusters", features=["x1", "x2"], target="reference", version=f"seed{seed}")
