"""Explicit metric definitions for classification, retrieval and clustering."""

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)


def classification_metrics(y_true: np.ndarray, prediction: np.ndarray,
                           probabilities: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(classes))
    report = classification_report(y_true, prediction, labels=labels, output_dict=True, zero_division=0)
    auc = {}
    for i, label in enumerate(classes):
        binary = (y_true == label).astype(int)
        if len(np.unique(binary)) == 2:
            auc[str(label)] = {"roc_auc_ovr": float(roc_auc_score(binary, probabilities[:, i])),
                               "average_precision_ovr": float(average_precision_score(binary, probabilities[:, i]))}
    return {"report": report, "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
            "per_class_auc": auc}


def retrieval_metrics(neighbors: np.ndarray, query_families: np.ndarray,
                      reference_families: np.ndarray, ks: list[int]) -> dict[str, Any]:
    """Binary family relevance. Report ranking on queries with relevant history.

    Recall@K denominator is ALL same-family references. AP@K denominator is
    min(K, relevant references). MRR is truncated at the largest requested K.
    """
    counts = Counter(reference_families)
    eligible = np.array([counts[family] > 0 for family in query_families])
    result: dict[str, Any] = {"eligible_queries": int(eligible.sum()),
                              "excluded_unseen_queries": int((~eligible).sum()), "by_k": {}}
    if not eligible.any():
        return result
    relevant = reference_families[neighbors[eligible]] == query_families[eligible, None]
    totals = np.array([counts[family] for family in query_families[eligible]])
    for requested_k in ks:
        k = min(requested_k, relevant.shape[1])
        rel = relevant[:, :k]
        ranks = np.arange(1, k + 1)
        precision_each_rank = np.cumsum(rel, axis=1) / ranks
        ideal_lengths = np.minimum(totals, k)
        discount = 1 / np.log2(ranks + 1)
        ideal_dcg = np.array([discount[:length].sum() for length in ideal_lengths])
        result["by_k"][str(requested_k)] = {
            "effective_k": k, "precision_at_k": float(rel.mean()),
            "recall_at_k": float(np.mean(rel.sum(axis=1) / totals)),
            "map_at_k": float(np.mean((precision_each_rank * rel).sum(axis=1) / ideal_lengths)),
            "ndcg_at_k": float(np.mean((rel * discount).sum(axis=1) / ideal_dcg)),
            "mrr_at_k": float(np.mean(np.max(rel / ranks, axis=1))),
        }
    return result


def novelty_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
    valid_auc = len(np.unique(truth)) == 2
    divide = lambda a, b: float(a / b) if b else None
    return {"threshold": float(threshold), "auroc": float(roc_auc_score(truth, scores)) if valid_auc else None,
            "auprc_average_precision": float(average_precision_score(truth, scores)) if valid_auc else None,
            "tpr": divide(tp, tp + fn), "fpr": divide(fp, fp + tn),
            "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "confusion_matrix_tn_fp_fn_tp": [int(v) for v in (tn, fp, fn, tp)],
            "known_count": int((~truth).sum()), "unseen_count": int(truth.sum()),
            "known_mean_score": float(scores[~truth].mean()) if (~truth).any() else None,
            "unseen_mean_score": float(scores[truth].mean()) if truth.any() else None}


def cluster_metrics(assignments: np.ndarray, families: np.ndarray) -> dict[str, Any]:
    """Weighted purity and unrelated pairs / all within-cluster pairs.

    Also report the fraction of non-singleton clusters with mixed families.
    Singletons contribute purity but no collapsed pairs.
    """
    pure_members = total_pairs = same_pairs = mixed_clusters = nonsingletons = 0
    for cluster in np.unique(assignments):
        labels = families[assignments == cluster]
        counts = Counter(labels)
        pure_members += max(counts.values())
        total_pairs += len(labels) * (len(labels) - 1) // 2
        same_pairs += sum(count * (count - 1) // 2 for count in counts.values())
        nonsingletons += len(labels) > 1
        mixed_clusters += len(counts) > 1
    cluster_count = len(np.unique(assignments))
    return {"original_alert_count": len(assignments), "cluster_count": cluster_count,
            "reduction_percentage": 100 * (1 - cluster_count / len(assignments)),
            "cluster_purity": pure_members / len(assignments),
            "false_merge_rate": (total_pairs - same_pairs) / total_pairs if total_pairs else 0.0,
            "within_cluster_pairs": total_pairs, "false_merge_pairs": total_pairs - same_pairs,
            "non_singleton_clusters": int(nonsingletons), "mixed_clusters": int(mixed_clusters),
            "mixed_cluster_rate": mixed_clusters / nonsingletons if nonsingletons else 0.0}
