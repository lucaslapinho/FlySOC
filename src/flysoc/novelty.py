"""Nearest-neighbor novelty with training-only leave-one-out calibration."""

import numpy as np
import pandas as pd
from scipy import sparse

from .similarity import SparseIndex


class NearestNeighborNovelty:
    def __init__(self, percentile: float = 99, metric: str = "jaccard", batch_size: int = 128):
        if not 0 < percentile <= 100:
            raise ValueError("Percentile must lie in (0, 100]")
        self.percentile, self.metric, self.batch_size = percentile, metric, batch_size

    def fit(self, reference: sparse.spmatrix, alert_ids: list[str] | None = None) -> "NearestNeighborNovelty":
        if reference.shape[0] < 2:
            raise ValueError("Novelty calibration requires at least two training alerts")
        self.index = SparseIndex(reference, self.metric, self.batch_size)
        self.alert_ids = alert_ids if alert_ids is not None else [str(i) for i in range(reference.shape[0])]
        if len(self.alert_ids) != reference.shape[0]:
            raise ValueError("Alert ID count differs from reference rows")
        scores, _ = self.index.query(reference, 1, exclude=np.arange(reference.shape[0]))
        self.calibration_scores_ = 1 - scores[:, 0]
        self.threshold_ = float(np.percentile(self.calibration_scores_, self.percentile))
        return self

    def score(self, queries: sparse.spmatrix) -> pd.DataFrame:
        similarities, indices = self.index.query(queries, 1)
        distances = 1 - similarities[:, 0]
        return pd.DataFrame({"novelty_score": distances, "is_novel": distances > self.threshold_,
                             "nearest_alert": [self.alert_ids[i] for i in indices[:, 0]],
                             "nearest_distance": distances})
