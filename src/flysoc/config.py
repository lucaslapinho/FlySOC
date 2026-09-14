"""Experiment configuration and validation."""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_FEATURES = {"verdict", "family", "is_unseen", "alert_id", "rule_id",
                      "mitre_technique", "mitre_tactic", "split", "metadata"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load safe YAML and reject inconsistent dimensions or leaking fields."""
    with Path(path or ROOT / "config/default.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    f, h, e = cfg["features"], cfg["flyhash"], cfg["evaluation"]
    ratios = [e[key] for key in ("train_ratio", "validation_ratio", "test_ratio")]
    if any(not 0 < value < 1 for value in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Temporal split ratios must be positive and sum to one")
    if f["input_features"] != f["categorical_hash_dim"] + f["text_max_features"] + 8:
        raise ValueError("input_features must equal categorical + text + 8 numeric features")
    for name in ("input_features", "categorical_hash_dim", "text_max_features"):
        if not isinstance(f[name], int) or f[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 1 <= h["fan_in"] <= f["input_features"]:
        raise ValueError("fan_in must be between 1 and input_features")
    if not 1 <= h["top_k"] <= h["kc_dim"] or h["kc_dim"] <= f["input_features"]:
        raise ValueError("Require expansion kc_dim > input_features and valid top_k")
    if not 0 < cfg["novelty"]["percentile"] <= 100:
        raise ValueError("Novelty percentile must be in (0, 100]")
    if not 0 < cfg["deduplication"]["similarity_threshold"] <= 1:
        raise ValueError("Deduplication threshold must be in (0, 1]")
    if h["batch_size"] < 1 or e["query_batch_size"] < 1:
        raise ValueError("Batch sizes must be positive")
    if not e["retrieval_k"] or min(e["retrieval_k"]) < 1:
        raise ValueError("Retrieval K must be positive")
    if not 0 < cfg["dataset"]["unseen_test_fraction"] < 1:
        raise ValueError("Unseen test fraction must be in (0, 1)")
    fields = sum([f[key] for key in ("text_fields", "categorical_fields", "identity_fields")], [])
    if FORBIDDEN_FEATURES.intersection(fields):
        raise ValueError("Label/evaluation/identifier fields cannot enter PN features")
    if any(f[key] < 0 for key in ("categorical_weight", "identity_weight", "text_weight", "numeric_weight")):
        raise ValueError("Feature weights must be nonnegative")
