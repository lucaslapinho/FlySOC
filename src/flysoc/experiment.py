"""Train all representations and baselines on chronological history only."""

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from .artifacts import environment_manifest, run_id, safe_run_path, sha256, write_json
from .baselines import classifiers, isolation_forest
from .config import ROOT, load_config
from .ingestion import chronological_split, read_alerts
from .novelty import NearestNeighborNovelty
from .pipeline import FlySOCPipeline
from .similarity import sparse_bytes

LOG = logging.getLogger(__name__)
VALID_LABELS = ["TRUE_POSITIVE", "FALSE_POSITIVE", "BENIGN"]


def train_experiment(dataset: Path, config_path: Path | None = None) -> str:
    cfg = load_config(config_path)
    frame = read_alerts(dataset)
    splits = chronological_split(frame, cfg["evaluation"]["train_ratio"], cfg["evaluation"]["validation_ratio"])
    heldout = cfg["dataset"]["heldout_family"]
    if "family" in frame:
        for name in ("train", "validation"):
            if (splits[name].family == heldout).any():
                raise ValueError(f"Held-out family occurs in {name}; novelty experiment would leak")
    identifier = run_id()
    output = safe_run_path(identifier)
    output.mkdir(parents=True, exist_ok=False)
    model_dir, processed_dir = ROOT / "models" / identifier, ROOT / "data/processed" / identifier
    model_dir.mkdir(parents=True, exist_ok=False)
    processed_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("metrics", "predictions", "plots"):
        (output / folder).mkdir()
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    LOG.info("Training %s: split sizes %s", identifier, {k: len(v) for k, v in splits.items()})
    start = perf_counter()
    pipeline = FlySOCPipeline(cfg)
    pipeline.fit(splits["train"])
    timing: dict[str, Any] = {"fly_pipeline_fit_seconds": perf_counter() - start, "classifier_fit_seconds": {}}
    pn, fp = pipeline.train_pn_, pipeline.train_fingerprints_
    LOG.info("PN %s (%s nonzeros), fingerprints %s, active KCs %s..%s", pn.shape, pn.nnz,
             fp.shape, fp.getnnz(axis=1).min(), fp.getnnz(axis=1).max())
    fit_models = {}
    train = splits["train"]
    labeled = train.verdict.isin(VALID_LABELS).to_numpy() if "verdict" in train else np.zeros(len(train), dtype=bool)
    if labeled.any() and train.loc[labeled, "verdict"].nunique() > 1:
        labels = train.loc[labeled, "verdict"].to_numpy(dtype=str)
        for representation, matrix in (("pn", pn), ("fly", fp)):
            for name, estimator in classifiers(cfg).items():
                key = f"{representation}_{name}"
                if name == "knn":
                    estimator.set_params(n_neighbors=min(cfg["classification"]["knn_neighbors"], len(labels)))
                start = perf_counter()
                estimator.fit(matrix[labeled], labels)
                timing["classifier_fit_seconds"][key] = perf_counter() - start
                fit_models[key] = estimator
                LOG.info("Fitted %s in %.3fs", key, timing["classifier_fit_seconds"][key])
    start = perf_counter()
    pn_novelty = NearestNeighborNovelty(cfg["novelty"]["percentile"], "cosine", cfg["evaluation"]["query_batch_size"])
    pn_novelty.fit(pn, train.alert_id.tolist())
    timing["pn_novelty_fit_seconds"] = perf_counter() - start
    start = perf_counter()
    isolation = isolation_forest(cfg)
    isolation.set_params(max_samples=min(isolation.max_samples, pn.shape[0])).fit(pn)
    # In-sample quantile is reported separately from leave-one-out NN calibration.
    isolation_threshold = float(np.percentile(-isolation.score_samples(pn), cfg["novelty"]["percentile"]))
    timing["isolation_forest_fit_seconds"] = perf_counter() - start
    payload = {"pipeline": pipeline, "classifiers": fit_models, "pn_novelty": pn_novelty,
               "isolation_forest": isolation, "isolation_threshold": isolation_threshold}
    model_path = model_dir / "experiment.joblib"
    joblib.dump(payload, model_path, compress=3)
    sparse.save_npz(processed_dir / "train_pn.npz", pn)
    sparse.save_npz(processed_dir / "train_fingerprints.npz", fp)
    for name, part in splits.items():
        part.to_csv(output / "predictions" / f"split_{name}.csv", index=False)
    manifest = {"run_id": identifier, "environment": environment_manifest(),
                "dataset_path": str(dataset.resolve()), "dataset_sha256": sha256(dataset),
                "dataset_size": len(frame), "model_sha256": sha256(model_path),
                "model_path": str(model_path), "configuration": cfg,
                "splits": {name: {"count": len(part), "start": str(part.timestamp.min()), "end": str(part.timestamp.max()),
                                   "verdict_counts": part.verdict.value_counts().to_dict() if "verdict" in part else {},
                                   "family_counts": part.family.value_counts().to_dict() if "family" in part else {}}
                           for name, part in splits.items()},
                "split_file_sha256": {name: sha256(output / "predictions" / f"split_{name}.csv") for name in splits}}
    write_json(output / "manifest.json", manifest)
    timing.update({"pn_shape": pn.shape, "fingerprint_shape": fp.shape,
                   "pn_storage_bytes": sparse_bytes(pn), "fingerprint_storage_bytes": sparse_bytes(fp),
                   "projection_storage_bytes": sparse_bytes(pipeline.encoder.projection_),
                   "model_file_bytes": model_path.stat().st_size,
                   "active_kcs_min": int(fp.getnnz(axis=1).min()), "active_kcs_max": int(fp.getnnz(axis=1).max()),
                   "fly_novelty_threshold": pipeline.novelty.threshold_, "pn_novelty_threshold": pn_novelty.threshold_})
    write_json(output / "metrics/training.json", timing)
    write_json(ROOT / "results/latest.json", {"run_id": identifier})
    LOG.info("Saved model and manifest: %s", output)
    return identifier


def load_experiment(identifier: str | None = None) -> tuple[Path, dict[str, Any], dict[str, pd.DataFrame]]:
    """Load only a local run with integrity checks. Joblib is a trusted-file format."""
    if identifier is None:
        identifier = json.loads((ROOT / "results/latest.json").read_text(encoding="utf-8"))["run_id"]
    output = safe_run_path(identifier)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    model_path = ROOT / "models" / identifier / "experiment.joblib"
    if sha256(model_path) != manifest["model_sha256"]:
        raise ValueError("Model artifact hash differs from manifest")
    splits = {}
    for name in ("train", "validation", "test"):
        path = output / "predictions" / f"split_{name}.csv"
        if sha256(path) != manifest["split_file_sha256"][name]:
            raise ValueError(f"Saved {name} split differs from manifest")
        splits[name] = read_alerts(path)
    return output, joblib.load(model_path), splits
