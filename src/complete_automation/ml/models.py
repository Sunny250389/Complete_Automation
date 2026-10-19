"""Model adapters: how the evaluator obtains predictions.

The evaluator only ever *reads* from a model: it calls ``predict`` and, when offered,
``predict_proba`` and ``decision_function``. It never trains, fits, saves or mutates one, and
it hands the model a copy of the features so a misbehaving model cannot alter the dataset.

Adapters provided:

* :class:`PrecomputedAdapter`: predictions that already exist (a file, a column, an array).
* :class:`EstimatorAdapter`: any in-memory object with ``predict`` (scikit-learn estimators and
  pipelines, or anything shaped like them). ``predict_proba`` and ``decision_function`` are used
  only when the object has them.
* :class:`FunctionAdapter`: plain callables, the simplest way to wrap a model-serving client.

**Extension point.** Subclass :class:`ModelAdapter` and implement ``predict`` (and optionally
``predict_proba`` / ``decision_function``) to evaluate models behind another mechanism. This
module deliberately loads no serialised model files: unpickling an untrusted file executes
arbitrary code. Load a trusted model in your own code and pass the object in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from complete_automation.ml.errors import MLConfigurationError, ModelInvocationError
from complete_automation.ml.schemas import Label, ModelInfo


@dataclass(frozen=True)
class Predictions:
    """What a model returned for a set of rows. Kept separate from evaluation results."""

    labels: np.ndarray
    """Output of ``predict``: class labels, numeric values or cluster ids."""
    probabilities: np.ndarray | None = None
    """``(n_rows, n_classes)``, columns ordered like ``classes``."""
    scores: np.ndarray | None = None
    """Output of ``decision_function``: ``(n_rows,)`` for binary problems, else ``(n_rows, n_classes)``."""
    classes: tuple[Label, ...] | None = None
    """Class order of the columns of ``probabilities`` and ``scores``."""


def as_array(raw: Any) -> np.ndarray:
    """``np.asarray`` that also flattens a single-column 2-D output such as ``(n, 1)``."""
    array = raw.to_numpy() if isinstance(raw, (pd.Series, pd.DataFrame)) else np.asarray(raw)
    if array.ndim == 2 and array.shape[1] == 1:
        return array.ravel()
    return array


def _python(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class ModelAdapter(ABC):
    """Uniform read-only view of a model."""

    info: ModelInfo
    supports_probabilities: bool = False
    supports_scores: bool = False
    expected_rows: int | None = None
    """Number of rows the adapter holds predictions for, when that is fixed (precomputed)."""
    can_predict_new_inputs: bool = True
    """``False`` for precomputed predictions: they cannot be recomputed on perturbed inputs."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> Any:
        """Labels, numeric values or cluster ids, one per row."""

    def predict_proba(self, features: pd.DataFrame) -> Any:
        raise NotImplementedError(f"{self.info.model_id} does not provide probabilities")

    def decision_function(self, features: pd.DataFrame) -> Any:
        raise NotImplementedError(f"{self.info.model_id} does not provide decision scores")

    @property
    def classes(self) -> tuple[Label, ...] | None:
        return None

    def collect(self, features: pd.DataFrame) -> Predictions:
        """Call the model on a copy of ``features`` and gather everything it offers."""
        try:
            labels = as_array(self.predict(features.copy()))
            probabilities = (
                np.asarray(self.predict_proba(features.copy()), dtype=float)
                if self.supports_probabilities
                else None
            )
            scores = (
                np.asarray(self.decision_function(features.copy()), dtype=float)
                if self.supports_scores
                else None
            )
        except MLConfigurationError:
            raise
        except Exception as error:  # noqa: BLE001 - any model failure is reported uniformly
            raise ModelInvocationError(
                f"model {self.info.model_id!r} failed while predicting: "
                f"{type(error).__name__}: {error}"
            ) from error
        return Predictions(labels, probabilities, scores, self.classes)


class PrecomputedAdapter(ModelAdapter):
    """Predictions computed elsewhere, aligned to dataset rows.

    ``labels`` may be a sequence (row ``i`` of the dataset gets ``labels[i]``) or a
    ``pandas.Series`` (aligned by its index, which is how rows removed by ``invalid_rows='exclude'``
    stay matched). ``probabilities`` and ``scores`` follow the position of ``labels``;
    probability columns are in ``classes`` order.
    """

    kind = "precomputed"
    can_predict_new_inputs = False

    def __init__(
        self,
        labels: Sequence[Any] | np.ndarray | pd.Series,
        *,
        probabilities: np.ndarray | pd.DataFrame | Sequence[Sequence[float]] | None = None,
        scores: np.ndarray | Sequence[Any] | None = None,
        classes: Sequence[Label] | None = None,
        model_id: str = "precomputed",
        version: str | None = None,
    ) -> None:
        self._labels = labels if isinstance(labels, pd.Series) else pd.Series(list(labels))
        if classes is None and isinstance(probabilities, pd.DataFrame):
            classes = tuple(probabilities.columns)
        if classes is None and (probabilities is not None or scores is not None):
            raise MLConfigurationError("probabilities or scores need the class order in `classes`")
        self._proba = None if probabilities is None else np.asarray(probabilities, dtype=float)
        self._scores = None if scores is None else np.asarray(scores, dtype=float)
        self._classes = None if classes is None else tuple(classes)
        self.supports_probabilities = self._proba is not None
        self.supports_scores = self._scores is not None
        self.expected_rows = len(self._labels)
        self.info = ModelInfo(model_id=model_id, version=version, kind=self.kind)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        prediction_column: str,
        *,
        probability_columns: Mapping[Label, str] | None = None,
        model_id: str = "precomputed",
        version: str | None = None,
    ) -> PrecomputedAdapter:
        """Use prediction columns that sit next to the features and labels in one table."""
        for column in [prediction_column, *(probability_columns or {}).values()]:
            if column not in frame.columns:
                raise MLConfigurationError(f"prediction column {column!r} is not in the dataset")
        probabilities = classes = None
        if probability_columns:
            classes = tuple(probability_columns)
            probabilities = frame[list(probability_columns.values())].to_numpy(dtype=float)
        return cls(
            frame[prediction_column],
            probabilities=probabilities,
            classes=classes,
            model_id=model_id,
            version=version,
        )

    @property
    def classes(self) -> tuple[Label, ...] | None:
        return self._classes

    def _rows(self, array: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Rows for ``features``; arrays of the wrong length are returned as they are (validation reports them)."""
        if len(array) != len(self._labels) or features.index.equals(self._labels.index):
            return array
        positions = self._labels.index.get_indexer(features.index)
        found = positions >= 0
        out = np.full((len(positions), *array.shape[1:]), np.nan)
        out[found] = array[positions[found]]
        return out

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._labels.index.equals(features.index):
            return self._labels.to_numpy()
        picked = self._labels.reindex(features.index)
        if picked.isna().any():
            return picked.to_numpy(dtype=object)
        return picked.astype(self._labels.dtype).to_numpy()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert self._proba is not None
        return self._rows(self._proba, features)

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        assert self._scores is not None
        return self._rows(self._scores, features)


class EstimatorAdapter(ModelAdapter):
    """Wrap an in-memory, already fitted estimator. Only ``predict*`` and ``decision_function`` are called."""

    kind = "estimator"

    def __init__(self, estimator: Any, *, model_id: str, version: str | None = None) -> None:
        if not callable(getattr(estimator, "predict", None)):
            raise MLConfigurationError(
                f"{type(estimator).__name__} has no callable predict(); wrap it with FunctionAdapter"
            )
        self._estimator = estimator
        self.supports_probabilities = callable(getattr(estimator, "predict_proba", None))
        self.supports_scores = callable(getattr(estimator, "decision_function", None))
        self.info = ModelInfo(
            model_id=model_id, version=version, kind=self.kind, class_name=type(estimator).__name__
        )

    @property
    def classes(self) -> tuple[Label, ...] | None:
        found = getattr(self._estimator, "classes_", None)
        return None if found is None else tuple(_python(c) for c in found)

    def predict(self, features: pd.DataFrame) -> Any:
        return self._estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> Any:
        return self._estimator.predict_proba(features)

    def decision_function(self, features: pd.DataFrame) -> Any:
        return self._estimator.decision_function(features)


class FunctionAdapter(ModelAdapter):
    """Wrap callables ``f(features_frame) -> array``, for example a call to a model server."""

    kind = "function"

    def __init__(
        self,
        predict: Callable[[pd.DataFrame], Any],
        *,
        model_id: str,
        version: str | None = None,
        predict_proba: Callable[[pd.DataFrame], Any] | None = None,
        decision_function: Callable[[pd.DataFrame], Any] | None = None,
        classes: Sequence[Label] | None = None,
    ) -> None:
        self._predict = predict
        self._predict_proba = predict_proba
        self._decision_function = decision_function
        self._classes = None if classes is None else tuple(classes)
        self.supports_probabilities = predict_proba is not None
        self.supports_scores = decision_function is not None
        self.info = ModelInfo(model_id=model_id, version=version, kind=self.kind)

    @property
    def classes(self) -> tuple[Label, ...] | None:
        return self._classes

    def predict(self, features: pd.DataFrame) -> Any:
        return self._predict(features)

    def predict_proba(self, features: pd.DataFrame) -> Any:
        assert self._predict_proba is not None
        return self._predict_proba(features)

    def decision_function(self, features: pd.DataFrame) -> Any:
        assert self._decision_function is not None
        return self._decision_function(features)
