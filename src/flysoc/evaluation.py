"""Four experiments with shared splits, saved predictions and explicit baselines."""

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import environment_manifest, run_id, write_json
from .clustering import deduplicate
from .experiment import VALID_LABELS, load_experiment
from .metrics import classification_metrics, cluster_metrics, novelty_metrics, retrieval_metrics
from .similarity import SparseIndex

LOG = logging.getLogger(__name__)


def latency_sample(index: SparseIndex, matrix: Any, k: int) -> dict[str, Any]:
    count = min(30, matrix.shape[0])
    if count == 0:
        return {}
    index.query(matrix[:1], k)  # warm up
    observations = []
    for i in np.linspace(0, matrix.shape[0] - 1, count, dtype=int):
        start = perf_counter()
        index.query(matrix[i], k)
        observations.append((perf_counter() - start) * 1000)
    return {"warm_queries": count, "median_ms": float(np.median(observations)),
            "p95_ms": float(np.percentile(observations, 95))}


def evaluate_experiment(identifier: str | None = None) -> Path:
    run_dir, payload, splits = load_experiment(identifier)
    output = run_dir / "evaluations" / run_id("eval")
    for name in ("metrics", "predictions", "plots"):
        (output / name).mkdir(parents=True, exist_ok=False)
    pipeline = payload["pipeline"]
    cfg = pipeline.config
    matrices = {"train": {"pn": pipeline.train_pn_, "fly": pipeline.train_fingerprints_}}
    performance: dict[str, Any] = {"encoding": {}, "retrieval": {}}
    for name in ("validation", "test"):
        start = perf_counter()
        pn = pipeline.preprocessor.transform(splits[name])
        pn_seconds = perf_counter() - start
        start = perf_counter()
        fp = pipeline.encoder.transform(pn)
        fly_seconds = perf_counter() - start
        matrices[name] = {"pn": pn, "fly": fp}
        performance["encoding"][name] = {"pn_seconds": pn_seconds, "fly_incremental_seconds": fly_seconds,
                                          "pn_ms_per_alert": 1000 * pn_seconds / len(pn.indptr[:-1]),
                                          "fly_end_to_end_ms_per_alert": 1000 * (pn_seconds + fly_seconds) / fp.shape[0],
                                          "active_kcs_min": int(fp.getnnz(axis=1).min()),
                                          "active_kcs_max": int(fp.getnnz(axis=1).max())}
    train, test = splits["train"], splits["test"]
    families_available = "family" in train and "family" in test and train.family.notna().all() and test.family.notna().all()
    unseen = ~test.family.isin(train.family).to_numpy() if families_available else None
    LOG.info("Encoded validation/test. Test known=%s, unseen=%s", int((~unseen).sum()) if unseen is not None else "unknown",
             int(unseen.sum()) if unseen is not None else "unknown")

    retrieval = {}
    neighbor_scores = {}
    for representation, metric in (("fly", "jaccard"), ("pn", "cosine")):
        index = SparseIndex(matrices["train"][representation], metric, cfg["evaluation"]["query_batch_size"])
        start = perf_counter()
        scores, indices = index.query(matrices["test"][representation], max(cfg["evaluation"]["retrieval_k"]))
        seconds = perf_counter() - start
        performance["retrieval"][representation] = {"batch_seconds": seconds,
                                                      "batch_ms_per_alert": seconds * 1000 / len(test),
                                                      "individual_queries": latency_sample(index, matrices["test"][representation], 5)}
        neighbor_scores[representation] = scores[:, 0]
        retrieval[representation] = retrieval_metrics(indices, test.family.to_numpy(), train.family.to_numpy(),
                                                       cfg["evaluation"]["retrieval_k"]) if families_available else {"status": "family ground truth unavailable"}
        rows = []
        for i in range(len(test)):
            for rank, (score, j) in enumerate(zip(scores[i], indices[i]), 1):
                rows.append({"query_alert_id": test.alert_id.iloc[i], "rank": rank,
                             "historical_alert_id": train.alert_id.iloc[j], "similarity": float(score),
                             "historical_verdict": train.verdict.iloc[j] if "verdict" in train else None,
                             "same_family": bool(test.family.iloc[i] == train.family.iloc[j]) if families_available else None})
        pd.DataFrame(rows).to_csv(output / "predictions" / f"retrieval_{representation}.csv", index=False)
    write_json(output / "metrics/retrieval.json", retrieval)
    pd.DataFrame([{"representation": name, "k": k, **values} for name, item in retrieval.items()
                  for k, values in item.get("by_k", {}).items()],
                 columns=["representation", "k", "effective_k", "precision_at_k", "recall_at_k",
                          "map_at_k", "ndcg_at_k", "mrr_at_k"]).to_csv(output / "metrics/retrieval.csv", index=False)
    LOG.info("Retrieval comparison generated")

    classification = {}
    class_rows = []
    for key, estimator in payload["classifiers"].items():
        representation = key.split("_", 1)[0]
        classification[key] = {}
        for part in ("validation", "test"):
            frame = splits[part]
            if "verdict" not in frame:
                continue
            labeled = frame.verdict.isin(VALID_LABELS).to_numpy()
            if not labeled.any():
                continue
            start = perf_counter()
            probabilities = estimator.predict_proba(matrices[part][representation][labeled])
            prediction = estimator.classes_[probabilities.argmax(axis=1)]
            y_true = frame.loc[labeled, "verdict"].to_numpy(dtype=str)
            result = classification_metrics(y_true, prediction, probabilities, estimator.classes_)
            result["prediction_seconds"] = perf_counter() - start
            classification[key][part] = result
            if part == "test" and unseen is not None:
                for scope, mask in (("test_known", ~unseen[labeled]), ("test_unseen", unseen[labeled])):
                    if mask.any():
                        classification[key][scope] = classification_metrics(y_true[mask], prediction[mask], probabilities[mask], estimator.classes_)
            records = pd.DataFrame({"alert_id": frame.loc[labeled, "alert_id"].to_numpy(), "actual_verdict": y_true,
                                    "predicted_verdict": prediction})
            for j, label in enumerate(estimator.classes_):
                records[f"probability_{label}"] = probabilities[:, j]
            records.to_csv(output / "predictions" / f"{key}_{part}.csv", index=False)
        for scope, item in classification[key].items():
            for label, row in item["report"].items():
                if isinstance(row, dict):
                    class_rows.append({"model": key, "split": scope, "class": label, **row})
    write_json(output / "metrics/classification.json", classification)
    pd.DataFrame(class_rows, columns=["model", "split", "class", "precision", "recall", "f1-score", "support"]).to_csv(output / "metrics/classification.csv", index=False)
    LOG.info("Compared %s downstream classification models", len(classification))

    novelty = {}
    distributions = {}
    for name, detector, representation in (("fly_nn", pipeline.novelty, "fly"), ("pn_nn", payload["pn_novelty"], "pn")):
        scores = detector.score(matrices["test"][representation])
        scores.insert(0, "alert_id", test.alert_id)
        if unseen is not None:
            scores["is_unseen_truth"] = unseen
            novelty[name] = novelty_metrics(unseen, scores.novelty_score.to_numpy(), detector.threshold_)
        else:
            novelty[name] = {"status": "family ground truth unavailable", "threshold": detector.threshold_}
        novelty[name]["calibration"] = "training leave-one-out nearest neighbor; strict score > percentile threshold"
        validation_scores = detector.score(matrices["validation"][representation])
        novelty[name]["validation_flag_rate"] = float(validation_scores.is_novel.mean())
        distributions[name] = scores.novelty_score.to_numpy()
        scores.to_csv(output / "predictions" / f"novelty_{name}.csv", index=False)
        validation_scores.to_csv(output / "predictions" / f"novelty_{name}_validation.csv", index=False)
    iso_scores = -payload["isolation_forest"].score_samples(matrices["test"]["pn"])
    iso_threshold = payload["isolation_threshold"]
    novelty["pn_isolation_forest"] = novelty_metrics(unseen, iso_scores, iso_threshold) if unseen is not None else {"status": "family ground truth unavailable"}
    novelty["pn_isolation_forest"]["calibration"] = "training in-sample score percentile"
    distributions["pn_isolation_forest"] = iso_scores
    pd.DataFrame({"alert_id": test.alert_id, "novelty_score": iso_scores, "is_novel": iso_scores > iso_threshold,
                  "is_unseen_truth": unseen if unseen is not None else pd.NA}).to_csv(output / "predictions/novelty_pn_isolation_forest.csv", index=False)
    write_json(output / "metrics/novelty.json", novelty)
    pd.DataFrame([{"model": name, **{k: v for k, v in item.items() if not isinstance(v, list)}}
                  for name, item in novelty.items()]).to_csv(output / "metrics/novelty.csv", index=False)
    LOG.info("Held-out novelty comparison generated")

    deduplication = {}
    for name, metric in (("fly", "jaccard"), ("pn", "cosine")):
        start = perf_counter()
        assignments = deduplicate(matrices["test"][name], cfg["deduplication"]["similarity_threshold"], metric)
        if families_available:
            deduplication[name] = cluster_metrics(assignments, test.family.to_numpy())
        else:
            deduplication[name] = {"original_alert_count": len(test), "cluster_count": len(np.unique(assignments)),
                                   "reduction_percentage": 100 * (1 - len(np.unique(assignments)) / len(test)),
                                   "purity_status": "family ground truth unavailable"}
        deduplication[name].update({"seconds": perf_counter() - start, "metric": metric,
                                    "threshold": cfg["deduplication"]["similarity_threshold"]})
        pd.DataFrame({"alert_id": test.alert_id, "cluster_id": assignments,
                      "family": test.family if families_available else pd.NA}).to_csv(output / "predictions" / f"clusters_{name}.csv", index=False)
    write_json(output / "metrics/deduplication.json", deduplication)
    pd.DataFrame([{"representation": name, **item} for name, item in deduplication.items()]).to_csv(output / "metrics/deduplication.csv", index=False)
    write_json(output / "metrics/performance.json", performance)
    all_metrics = {"retrieval": retrieval, "classification": classification, "novelty": novelty,
                   "deduplication": deduplication, "performance": performance}
    write_json(output / "metrics/summary.json", all_metrics)
    write_json(output / "environment.json", environment_manifest())
    from .plotting import create_plots
    create_plots(output / "plots", all_metrics, neighbor_scores, distributions, unseen)
    write_report(output, all_metrics, len(train), len(test))
    write_json(run_dir / "latest_evaluation.json", {"evaluation_id": output.name})
    LOG.info("Evaluation complete: %s", output)
    return output


def write_report(output: Path, metrics: dict[str, Any], train_count: int, test_count: int) -> None:
    lines = ["# FlySOC experiment report", "", f"Training alerts: {train_count}; test alerts: {test_count}.", "",
             "All results below were computed from saved chronological splits. Validation was evaluated separately; no test-driven tuning was performed.", "",
             "## Retrieval", "", "Relevance means identical synthetic operational family. Queries without relevant training history are excluded from ranking metrics.", "",
             "| Representation | P@5 | Recall@5 | MRR@10 |", "|---|---:|---:|---:|"]
    for name, item in metrics["retrieval"].items():
        by_k = item.get("by_k", {})
        if "5" in by_k and "10" in by_k:
            lines.append(f"| {name} | {by_k['5']['precision_at_k']:.4f} | {by_k['5']['recall_at_k']:.4f} | {by_k['10']['mrr_at_k']:.4f} |")
    lines += ["", "## Verdict classification", "", "These are explicit downstream classifiers. Scores include the held-out family in the full test period.", "",
              "| Model | Test macro F1 | Known-only macro F1 |", "|---|---:|---:|"]
    for name, item in metrics["classification"].items():
        if "test" in item:
            known = item.get("test_known", {}).get("report", {}).get("macro avg", {}).get("f1-score")
            lines.append(f"| {name} | {item['test']['report']['macro avg']['f1-score']:.4f} | {f'{known:.4f}' if known is not None else 'N/A'} |")
    lines += ["", "## Unseen-family novelty", "", "AUPRC is average precision. Flags use a strict score > threshold comparison. Novelty is not evidence of maliciousness.", "",
              "| Model | AUROC | AUPRC | TPR | FPR |", "|---|---:|---:|---:|---:|"]
    for name, item in metrics["novelty"].items():
        if item.get("auroc") is not None:
            lines.append(f"| {name} | {item['auroc']:.4f} | {item['auprc_average_precision']:.4f} | {item['tpr']:.4f} | {item['fpr']:.4f} |")
    lines += ["", "## Deduplication", "", "The same numerical threshold has different meanings for Jaccard and cosine. This comparison is descriptive, not a matched-operating-point comparison.", "",
              "| Representation | Clusters | Reduction | Purity | False merge rate |", "|---|---:|---:|---:|---:|"]
    for name, item in metrics["deduplication"].items():
        if "cluster_purity" in item:
            lines.append(f"| {name} | {item['cluster_count']} | {item['reduction_percentage']:.2f}% | {item['cluster_purity']:.4f} | {item['false_merge_rate']:.4f} |")
    lines += ["", "## Interpretation limits", "",
              "One deterministic synthetic run establishes an executable research baseline, not deployment readiness or superiority. Repeated templates can make retrieval easy even with temporal splitting. Ground-truth families are coarse operational proxies; high purity does not establish incident-level deduplication safety. Verdict noise is synthetic. Train-only calibration may drift in later periods. Zero-input fingerprints are explicitly empty; distances alone cannot distinguish missing telemetry from novelty.", "",
              "Next experiment: repeat across multiple seeds, vary fan-in/top-k, and use analyst-reviewed relevance pairs from de-identified CSV exports with temporal and campaign separation. Select deduplication thresholds on validation at a fixed false-merge budget before testing."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
