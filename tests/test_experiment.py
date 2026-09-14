import json

import pandas as pd
import yaml

from flysoc import experiment
from flysoc.evaluation import evaluate_experiment


def test_file_export_without_optional_columns_completes(config, alerts, tmp_path, monkeypatch):
    """An unannotated export still yields usable scores, not invented metrics."""
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "safe_run_path", lambda identifier: tmp_path / "results" / identifier)
    csv_path = tmp_path / "minimal_export.csv"
    alerts[["timestamp", "alert_name"]].to_csv(csv_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    identifier = experiment.train_experiment(csv_path, config_path)
    output = evaluate_experiment(identifier)
    result = json.loads((output / "metrics/summary.json").read_text())
    assert result["classification"] == {}
    assert result["retrieval"]["fly"]["status"] == "family ground truth unavailable"
    assert "auroc" not in result["novelty"]["fly_nn"]
    assert pd.read_csv(output / "metrics/classification.csv").empty
    assert pd.read_csv(output / "metrics/retrieval.csv").empty
    assert len(pd.read_csv(output / "predictions/novelty_fly_nn.csv")) == 45
