"""Reusable representation, memory and novelty pipeline."""

from typing import Any

import pandas as pd
from scipy import sparse

from .config import validate_config
from .flyhash import FlyHash
from .memory import AssociativeMemory
from .novelty import NearestNeighborNovelty
from .preprocessing import AlertPreprocessor


class FlySOCPipeline:
    def __init__(self, config: dict[str, Any]):
        validate_config(config)
        self.config = config
        self.preprocessor = AlertPreprocessor(config["features"])
        self.encoder = FlyHash(**config["flyhash"], random_seed=config["seed"])
        self.memory = AssociativeMemory(config["evaluation"]["query_batch_size"])
        self.novelty = NearestNeighborNovelty(config["novelty"]["percentile"],
                                             batch_size=config["evaluation"]["query_batch_size"])

    def fit(self, train: pd.DataFrame) -> "FlySOCPipeline":
        self.train_pn_ = self.preprocessor.fit_transform(train)
        self.train_fingerprints_ = self.encoder.fit_transform(self.train_pn_)
        self.memory.fit(self.train_fingerprints_, train)
        self.novelty.fit(self.train_fingerprints_, train.alert_id.astype(str).tolist())
        return self

    def encode(self, alerts: pd.DataFrame) -> sparse.csr_matrix:
        return self.encoder.transform(self.preprocessor.transform(alerts))

    def inspect(self, alerts: pd.DataFrame, k: int = 5) -> list[dict[str, Any]]:
        fingerprints = self.encode(alerts)
        matches = self.memory.retrieve(fingerprints, k)
        novelty = self.novelty.score(fingerprints)
        return [{"alert_id": row.get("alert_id"), "alert_name": row.get("alert_name"),
                 "kc_dim": fingerprints.shape[1], "active_kcs": int(fingerprints[i].nnz),
                 "active_indices": fingerprints[i].indices.tolist(),
                 "novelty_threshold": self.novelty.threshold_, **novelty.iloc[i].to_dict(),
                 "historical_matches": matches[i]}
                for i, (_, row) in enumerate(alerts.iterrows())]
