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
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
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

    representation = cfg.get("representation", {})
    backend = representation.get("backend", "flyhash")
    supported = {"flyhash", "connectome", "rewired_connectome"}
    if backend not in supported:
        raise ValueError(f"representation.backend must be one of {sorted(supported)}")

    connectome = cfg.get("connectome", {})
    if backend != "flyhash":
        if not connectome.get("enabled", False):
            raise ValueError("Connectome backends require connectome.enabled=true")
        if not connectome.get("projection_path"):
            raise ValueError("Connectome backends require connectome.projection_path")
    for name in ("bridge_fan_out", "top_k", "batch_size"):
        if name in connectome and (not isinstance(connectome[name], int) or connectome[name] < 1):
            raise ValueError(f"connectome.{name} must be a positive integer")

    rewired = cfg.get("controls", {}).get("rewired", {})
    if backend == "rewired_connectome" and not rewired.get("enabled", False):
        raise ValueError("rewired_connectome requires controls.rewired.enabled=true")
    if backend == "rewired_connectome" and not rewired.get("preserve_degree", True):
        raise ValueError("rewired_connectome requires degree preservation")
    if rewired.get("swaps_per_edge", 0) < 0:
        raise ValueError("controls.rewired.swaps_per_edge must be nonnegative")

    ablation = f.get("ablation", {})
    disabled = ablation.get("disabled_fields", [])
    if not isinstance(disabled, list) or any(not isinstance(name, str) for name in disabled):
        raise ValueError("features.ablation.disabled_fields must be a list of field names")
    if FORBIDDEN_FEATURES.intersection(disabled):
        raise ValueError("Ablation may only target configured input fields")
    allowed_ablation_fields = set(fields) | {"source_port", "destination_port", "timestamp"}
    unknown_disabled = set(disabled) - allowed_ablation_fields
    if unknown_disabled:
        raise ValueError(f"Unknown feature-ablation fields: {sorted(unknown_disabled)}")

    seeds = cfg.get("experiments", {}).get("seeds", [cfg["seed"]])
    if not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("experiments.seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiments.seeds must be unique")

    mbon = cfg.get("mbon", {})
    if mbon:
        if mbon.get("learning_rate", 0) <= 0:
            raise ValueError("mbon.learning_rate must be positive")
        if not 0 <= mbon.get("weight_decay", 0) < 1:
            raise ValueError("mbon.weight_decay must be in [0, 1)")
        if mbon.get("weight_clip", 0) <= 0:
            raise ValueError("mbon.weight_clip must be positive")

    max_false_merge_rate = cfg.get("deduplication", {}).get("max_false_merge_rate", 0.01)
    if not 0 <= max_false_merge_rate <= 1:
        raise ValueError("deduplication.max_false_merge_rate must be in [0, 1]")
    thresholds = cfg.get("deduplication", {}).get("thresholds", {})
    if any(value is not None and not 0 < value <= 1 for value in thresholds.values()):
        raise ValueError("Representation-specific deduplication thresholds must be null or in (0, 1]")
