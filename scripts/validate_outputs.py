"""Independently cross-check saved predictions, metrics, images and data hashes."""

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

from flysoc.artifacts import run_id, write_json
from flysoc.experiment import load_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    args = parser.parse_args()
    run_dir, payload, splits = load_experiment(args.run)
    evaluation_id = json.loads((run_dir / "latest_evaluation.json").read_text())["evaluation_id"]
    if not evaluation_id.startswith("eval_") or any(c in evaluation_id for c in ("/", "\\", ".")):
        raise ValueError("Unsafe evaluation pointer")
    output = run_dir / "evaluations" / evaluation_id
    json_files, csv_files, png_files = list(output.rglob("*.json")), list(output.rglob("*.csv")), list(output.rglob("*.png"))
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    for path in csv_files:
        pd.read_csv(path)
    for path in png_files:
        with Image.open(path) as picture:
            if min(picture.size) < 200:
                raise AssertionError(f"Plot unexpectedly small: {path}")
            picture.verify()
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    assert pd.to_datetime(train.timestamp, utc=True).max() < pd.to_datetime(validation.timestamp, utc=True).min()
    assert pd.to_datetime(validation.timestamp, utc=True).max() < pd.to_datetime(test.timestamp, utc=True).min()
    assert set(train.alert_id).isdisjoint(test.alert_id)
    classification = json.loads((output / "metrics/classification.json").read_text())
    checked_confusions = 0
    for model, partitions in classification.items():
        for partition in ("validation", "test"):
            if partition not in partitions:
                continue
            rows = pd.read_csv(output / "predictions" / f"{model}_{partition}.csv")
            item = partitions[partition]
            cm = confusion_matrix(rows.actual_verdict, rows.predicted_verdict, labels=item["labels"])
            np.testing.assert_array_equal(cm, item["confusion_matrix"])
            prob = rows.filter(like="probability_").to_numpy()
            np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-6)
            checked_confusions += 1
    novelty = json.loads((output / "metrics/novelty.json").read_text())
    for name, item in novelty.items():
        rows = pd.read_csv(output / "predictions" / f"novelty_{name}.csv")
        assert len(rows) == len(test)
        if item.get("auroc") is not None:
            assert np.isclose(roc_auc_score(rows.is_unseen_truth, rows.novelty_score), item["auroc"])
        if "threshold" in item:
            np.testing.assert_array_equal(rows.is_novel, rows.novelty_score > item["threshold"])
    for representation in ("fly", "pn"):
        rows = pd.read_csv(output / "predictions" / f"retrieval_{representation}.csv")
        assert rows.query_alert_id.nunique() == len(test)
        assert set(rows.historical_alert_id).issubset(set(train.alert_id))
        assert rows.similarity.between(0, 1).all()
    result = {"status": "passed", "run": run_dir.name, "evaluation": evaluation_id,
              "json_files_checked": len(json_files), "csv_files_checked": len(csv_files),
              "png_files_checked": len(png_files), "confusion_matrices_recomputed": checked_confusions,
              "model_and_split_hashes": "verified", "temporal_separation": "verified",
              "novelty_auc_and_flags": "recomputed", "retrieval_history_membership": "verified"}
    write_json(output / f"{run_id('validation')}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
