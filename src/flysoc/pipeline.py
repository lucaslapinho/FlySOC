"""Reusable representation, memory and novelty pipeline."""

from typing import Any
from pathlib import Path

import pandas as pd
from scipy import sparse

from .config import ROOT, validate_config
from .flyhash import FlyHash
from .memory import AssociativeMemory
from .novelty import NearestNeighborNovelty
from .preprocessing import AlertPreprocessor


def _artifact_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_encoder(config: dict[str, Any]):
    """Build the configured sparse fingerprint backend without changing defaults."""
    backend = config.get("representation", {}).get("backend", "flyhash")
    if backend == "flyhash":
        return FlyHash(**config["flyhash"], random_seed=config["seed"])

    from .connectome_projection import ConnectomeHash, load_connectome_projection

    settings = config["connectome"]
    loaded = load_connectome_projection(_artifact_path(settings["projection_path"]))
    if loaded is None:  # required=True in the loader makes this defensive only
        raise FileNotFoundError("Configured connectome projection is unavailable")
    projection, provenance = loaded
    if backend == "rewired_connectome":
        from .projection_controls import degree_preserving_rewire

        control = config["controls"]["rewired"]
        projection = degree_preserving_rewire(
            projection,
            swaps_per_edge=control["swaps_per_edge"],
            random_seed=config["seed"],
        )
        provenance = {
            **provenance,
            "experimental_control": {
                "kind": "degree_preserving_rewire",
                "seed": config["seed"],
                **projection.rewire_stats_,
            },
        }

    encoder = ConnectomeHash(
        projection,
        input_dim=config["features"]["input_features"],
        top_k=settings["top_k"],
        bridge_fan_out=settings["bridge_fan_out"],
        random_seed=config["seed"],
        batch_size=settings["batch_size"],
        provenance=provenance,
    )
    encoder.backend_name = backend
    return encoder


class FlySOCPipeline:
    def __init__(self, config: dict[str, Any]):
        validate_config(config)
        self.config = config
        self.preprocessor = AlertPreprocessor(config["features"])
        self.encoder = build_encoder(config)
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
