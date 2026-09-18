"""Validation-only operating-point selection for deduplication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse

from .clustering import deduplicate


def _cluster_stats(assignments: np.ndarray, families: np.ndarray) -> dict[str, float | int]:
    total_pairs = 0
    false_pairs = 0
    for cluster in np.unique(assignments):
        labels = families[assignments == cluster]
        if len(labels) < 2:
            continue
        cluster_pairs = len(labels) * (len(labels) - 1) // 2
        _, counts = np.unique(labels, return_counts=True)
        same_family_pairs = int(sum(count * (count - 1) // 2 for count in counts))
        total_pairs += cluster_pairs
        false_pairs += cluster_pairs - same_family_pairs

    cluster_count = len(np.unique(assignments))
    return {
        "clusters": int(cluster_count),
        "reduction": 1.0 - cluster_count / len(assignments),
        "within_cluster_pairs": int(total_pairs),
        "false_merge_pairs": int(false_pairs),
        "false_merge_rate": false_pairs / total_pairs if total_pairs else 0.0,
    }


def calibrate_dedup_threshold(
    values: sparse.spmatrix,
    families: Sequence[Any],
    metric: str,
    max_false_merge_rate: float = 0.01,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Select maximum validation reduction within a false-merge budget.

    Calibration is for exactly one representation and metric.  The false-merge
    rate is the fraction of merged within-cluster pairs whose family labels
    disagree.  Ties in reduction prefer the higher, more conservative
    threshold.
    """
    matrix = sparse.csr_matrix(values)
    labels = np.asarray(families, dtype=object)
    if matrix.shape[0] < 2 or labels.shape != (matrix.shape[0],):
        raise ValueError("At least two aligned validation family labels are required")
    if metric not in {"jaccard", "cosine"}:
        raise ValueError("metric must be 'jaccard' or 'cosine'")
    if not 0 <= max_false_merge_rate <= 1:
        raise ValueError("max_false_merge_rate must lie in [0, 1]")

    candidate_values = np.linspace(0.50, 0.99, 50) if thresholds is None else thresholds
    candidate_thresholds = sorted({float(value) for value in candidate_values})
    if not candidate_thresholds or any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in candidate_thresholds
    ):
        raise ValueError("thresholds must contain finite values in (0, 1]")

    candidates = []
    for threshold in candidate_thresholds:
        assignments = deduplicate(matrix, threshold=threshold, metric=metric)
        candidates.append(
            {"threshold": threshold, **_cluster_stats(assignments, labels)}
        )

    feasible = [
        row
        for row in candidates
        if row["false_merge_rate"] <= max_false_merge_rate
    ]
    if feasible:
        selected = max(feasible, key=lambda row: (row["reduction"], row["threshold"]))
        status = "budget_met"
    else:
        selected = min(
            candidates,
            key=lambda row: (row["false_merge_rate"], -row["threshold"]),
        )
        status = "no_threshold_met_budget"

    return {
        "status": status,
        "metric": metric,
        "max_false_merge_rate": max_false_merge_rate,
        "selected": selected,
        "candidates": candidates,
    }


def calibrate_thresholds(
    representations: Mapping[str, sparse.spmatrix],
    families: Sequence[Any] | Mapping[str, Sequence[Any]],
    metrics_by_representation: Mapping[str, Sequence[str]],
    max_false_merge_rate: float = 0.01,
    thresholds: Sequence[float] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Calibrate each representation/metric pair independently on validation.

    ``families`` may be one shared aligned label vector or a mapping when the
    representations have distinct validation rows.  No selected threshold is
    reused across representations or similarity metrics.
    """
    if not representations:
        raise ValueError("At least one representation is required")
    unknown = set(metrics_by_representation) - set(representations)
    missing = set(representations) - set(metrics_by_representation)
    if unknown or missing:
        raise ValueError("Representations and metric configuration must have identical keys")

    results: dict[str, dict[str, dict[str, Any]]] = {}
    labels_are_mapped = isinstance(families, Mapping)
    if labels_are_mapped and set(families) != set(representations):
        raise ValueError("Mapped family labels must match all representations")

    for representation, matrix in representations.items():
        metrics = list(metrics_by_representation[representation])
        if not metrics:
            raise ValueError(f"No metrics configured for representation {representation!r}")
        labels = families[representation] if labels_are_mapped else families
        results[representation] = {
            metric: calibrate_dedup_threshold(
                matrix,
                labels,
                metric=metric,
                max_false_merge_rate=max_false_merge_rate,
                thresholds=thresholds,
            )
            for metric in metrics
        }
    return results
