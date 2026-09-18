"""Validation-calibrated conventional novelty baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _validate_partition(values: Any, name: str, minimum_rows: int):
    matrix = sparse.csr_matrix(values) if sparse.issparse(values) else np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] < minimum_rows or matrix.shape[1] < 1:
        raise ValueError(
            f"{name} must be a two-dimensional matrix with at least "
            f"{minimum_rows} rows and one feature"
        )
    finite_values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(finite_values).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


class ValidationCalibratedNoveltyBaselines:
    """Fit LOF and One-Class SVM on training data and calibrate on validation.

    Both scores are oriented so larger values mean more anomalous.  The
    validation partition is expected to contain in-distribution examples; its
    selected percentile is the empirical false-positive operating point.  Test
    labels and test scores are never consumed by :meth:`fit`.
    """

    def __init__(
        self,
        percentile: float = 99,
        lof_neighbors: int = 35,
        ocsvm_nu: float = 0.01,
        ocsvm_gamma: str | float = "scale",
    ):
        if not 0 < percentile <= 100:
            raise ValueError("percentile must lie in (0, 100]")
        if lof_neighbors < 1:
            raise ValueError("lof_neighbors must be positive")
        if not 0 < ocsvm_nu <= 1:
            raise ValueError("ocsvm_nu must lie in (0, 1]")
        self.percentile = float(percentile)
        self.lof_neighbors = int(lof_neighbors)
        self.ocsvm_nu = float(ocsvm_nu)
        self.ocsvm_gamma = ocsvm_gamma

    def fit(self, train: Any, validation: Any) -> "ValidationCalibratedNoveltyBaselines":
        train = _validate_partition(train, "train", minimum_rows=3)
        validation = _validate_partition(validation, "validation", minimum_rows=1)
        if train.shape[1] != validation.shape[1]:
            raise ValueError("Training and validation feature widths differ")

        effective_neighbors = min(self.lof_neighbors, train.shape[0] - 1)
        self.lof_ = LocalOutlierFactor(
            n_neighbors=effective_neighbors,
            novelty=True,
            n_jobs=1,
        )
        self.ocsvm_ = OneClassSVM(
            kernel="rbf",
            gamma=self.ocsvm_gamma,
            nu=self.ocsvm_nu,
        )
        self.lof_.fit(train)
        self.ocsvm_.fit(train)

        validation_scores = {
            "lof": -self.lof_.score_samples(validation),
            "ocsvm": -self.ocsvm_.score_samples(validation),
        }
        self.thresholds_ = {
            name: float(np.percentile(scores, self.percentile))
            for name, scores in validation_scores.items()
        }
        self.calibration_scores_ = validation_scores
        self.n_features_in_ = train.shape[1]
        self.effective_lof_neighbors_ = effective_neighbors
        return self

    def score(self, values: Any) -> dict[str, dict[str, np.ndarray | float]]:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("Fit novelty baselines before scoring")
        values = _validate_partition(values, "values", minimum_rows=1)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Scoring feature width differs from fitted data")

        scores = {
            "lof": -self.lof_.score_samples(values),
            "ocsvm": -self.ocsvm_.score_samples(values),
        }
        return {
            name: {
                "score": model_scores,
                "threshold": self.thresholds_[name],
                "is_novel": model_scores > self.thresholds_[name],
            }
            for name, model_scores in scores.items()
        }
