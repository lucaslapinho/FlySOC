"""Deferred benchmark: compare PN, FlyHash, measured and rewired projections.

This script is intentionally not invoked during installation or preparation.
It exists for a later controlled experiment with local data and a verified
MaleCNS-derived projection artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from flysoc.connectome_projection import ConnectomeHash, load_connectome_projection
from flysoc.config import load_config
from flysoc.flyhash import FlyHash
from flysoc.ingestion import chronological_split, read_alerts
from flysoc.novelty import NearestNeighborNovelty
from flysoc.preprocessing import AlertPreprocessor
from flysoc.projection_controls import degree_preserving_rewire, degree_signature
from flysoc.similarity import SparseIndex, sparse_bytes


def retrieval_precision_at_five(train, test, train_family, test_family, metric: str):
    index = SparseIndex(train, metric=metric, batch_size=128)
    _, neighbors = index.query(test, 5)
    eligible = np.isin(test_family, np.unique(train_family))
    if not eligible.any():
        return None
    hits = [
        (train_family[neighbors[row]] == test_family[row]).mean()
        for row in np.flatnonzero(eligible)
    ]
    return float(np.mean(hits))


def summarize_representation(name, train, test, train_ids, metric: str):
    started = perf_counter()
    detector = NearestNeighborNovelty(99, metric=metric, batch_size=128)
    detector.fit(train, train_ids)
    detector.score(test)
    return {
        "representation": name,
        "train_shape": list(train.shape),
        "train_nnz": int(train.nnz),
        "storage_bytes": sparse_bytes(train),
        "active_mean": float(np.mean(train.getnnz(axis=1))),
        "novelty_fit_and_score_seconds": perf_counter() - started,
        "novelty_threshold": float(detector.threshold_),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", help="Comma-separated override; default config experiments.seeds")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else cfg.get("experiments", {}).get("seeds", [cfg["seed"]])
    )
    loaded = load_connectome_projection(args.projection)
    if loaded is None:
        raise FileNotFoundError("Connectome projection is unavailable")
    projection, provenance = loaded

    frame = read_alerts(args.data)
    splits = chronological_split(
        frame,
        cfg["evaluation"]["train_ratio"],
        cfg["evaluation"]["validation_ratio"],
    )
    train, test = splits["train"], splits["test"]
    preprocessor = AlertPreprocessor(cfg["features"]).fit(train)
    train_pn, test_pn = preprocessor.transform(train), preprocessor.transform(test)
    train_ids = train.alert_id.astype(str).tolist()
    has_families = "family" in train and "family" in test
    train_family = train.family.to_numpy(dtype=str) if has_families else None
    test_family = test.family.to_numpy(dtype=str) if has_families else None

    rows = []
    pn_row = summarize_representation("pn", train_pn, test_pn, train_ids, "cosine")
    if has_families:
        pn_row["retrieval_precision_at_5"] = retrieval_precision_at_five(
            train_pn, test_pn, train_family, test_family, "cosine"
        )
    rows.append({"seed": None, **pn_row})

    top_k = cfg["connectome"]["top_k"]
    bridge_fan_out = cfg["connectome"]["bridge_fan_out"]
    batch_size = cfg["connectome"]["batch_size"]
    for seed in seeds:
        flyhash = FlyHash(**cfg["flyhash"], random_seed=seed).fit(train_pn)
        rewired_projection = degree_preserving_rewire(
            projection,
            cfg["controls"]["rewired"]["swaps_per_edge"],
            seed,
        )
        before, after = degree_signature(projection), degree_signature(rewired_projection)
        if not all(np.array_equal(left, right) for left, right in zip(before, after)):
            raise RuntimeError("Rewired control changed the bipartite degree signature")

        encoders = {
            "flyhash": flyhash,
            "connectome": ConnectomeHash(
                projection, train_pn.shape[1], top_k, bridge_fan_out, seed, batch_size,
                provenance=provenance,
            ),
            "rewired_connectome": ConnectomeHash(
                rewired_projection, train_pn.shape[1], top_k, bridge_fan_out, seed,
                batch_size, provenance={**provenance, "control": "degree_preserving_rewire"},
            ),
        }
        for name, encoder in encoders.items():
            encoded_train = encoder.transform(train_pn)
            encoded_test = encoder.transform(test_pn)
            row = summarize_representation(name, encoded_train, encoded_test, train_ids, "jaccard")
            if has_families:
                row["retrieval_precision_at_5"] = retrieval_precision_at_five(
                    encoded_train, encoded_test, train_family, test_family, "jaccard"
                )
            rows.append({"seed": seed, **row})

    args.output.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "results.csv", index=False)
    aggregations: dict[str, str | list[str]] = {
        "storage_bytes": "mean",
        "active_mean": "mean",
    }
    if has_families:
        aggregations["retrieval_precision_at_5"] = ["mean", "std"]
    summary = results.groupby("representation", dropna=False).agg(aggregations)
    summary.to_csv(args.output / "summary.csv")
    manifest = {
        "data": str(args.data.resolve()),
        "projection": str(args.projection.resolve()),
        "projection_provenance": provenance,
        "seeds": seeds,
        "scientific_note": (
            "Measured and rewired projections use identical feature bridges; "
            "the intended manipulated variable is PN-to-KC topology."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
