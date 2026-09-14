"""Historical associative lookup with auditable analyst label corrections."""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from scipy import sparse

from .similarity import SparseIndex, binary_matrix

VERDICTS = {"TRUE_POSITIVE", "FALSE_POSITIVE", "BENIGN", "UNKNOWN"}


class AssociativeMemory:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.feedback_log: list[dict[str, Any]] = []

    def fit(self, fingerprints: sparse.spmatrix, metadata: pd.DataFrame) -> "AssociativeMemory":
        if len(metadata) != fingerprints.shape[0] or "alert_id" not in metadata:
            raise ValueError("Aligned metadata with alert_id is required")
        if metadata.alert_id.isna().any() or metadata.alert_id.astype(str).duplicated().any():
            raise ValueError("Historical alert IDs must be unique and non-null")
        self.fingerprints = binary_matrix(fingerprints)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.metadata["alert_id"] = self.metadata.alert_id.astype(str)
        self.index = SparseIndex(self.fingerprints, metric="jaccard", batch_size=self.batch_size)
        return self

    def retrieve(self, query: sparse.spmatrix, k: int = 5) -> list[list[dict[str, Any]]]:
        scores, indices = self.index.query(query, k)
        return [[{**self.metadata.iloc[int(index)].to_dict(), "similarity": float(score)}
                 for score, index in zip(row_scores, row_indices)]
                for row_scores, row_indices in zip(scores, indices)]

    def add(self, fingerprints: sparse.spmatrix, metadata: pd.DataFrame) -> None:
        """Append historical alerts; novelty calibration remains explicitly frozen."""
        combined = sparse.vstack([self.fingerprints, binary_matrix(fingerprints)], format="csr")
        self.fit(combined, pd.concat([self.metadata, metadata], ignore_index=True))

    def update_feedback(self, alert_id: str, verdict: str, note: str = "") -> None:
        """Correct stored verdicts; no automatic downstream classifier retraining."""
        if verdict not in VERDICTS:
            raise ValueError("Unsupported verdict")
        matches = self.metadata.index[self.metadata.alert_id == str(alert_id)]
        if len(matches) != 1:
            raise KeyError(alert_id)
        index = matches[0]
        old = self.metadata.loc[index, "verdict"] if "verdict" in self.metadata else None
        self.metadata.loc[index, "verdict"] = verdict
        self.feedback_log.append({"alert_id": str(alert_id), "previous_verdict": old,
                                  "verdict": verdict, "note": note,
                                  "timestamp": datetime.now(timezone.utc).isoformat()})
