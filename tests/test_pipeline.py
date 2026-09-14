import joblib
import numpy as np
import pandas as pd
import pytest

from flysoc.ingestion import chronological_split, read_alerts
from flysoc.pipeline import FlySOCPipeline
from flysoc.synthetic import generate_alerts


def test_end_to_end_and_serialization(config, alerts, tmp_path):
    split = chronological_split(alerts)
    model = FlySOCPipeline(config).fit(split["train"])
    test = split["test"].head(3)
    encoded = model.encode(test)
    assert encoded.shape == (3, 256)
    assert np.all(encoded.getnnz(axis=1) == 16)
    result = model.inspect(test, 3)
    assert len(result) == 3 and len(result[0]["historical_matches"]) == 3
    assert 0 <= result[0]["novelty_score"] <= 1
    path = tmp_path / "trusted.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert (encoded != restored.encode(test)).nnz == 0
    assert restored.inspect(test, 3)[0]["nearest_alert"] == result[0]["nearest_alert"]


def test_temporal_split_and_family_holdout(config, alerts):
    split = chronological_split(alerts.sample(frac=1, random_state=5))
    assert [len(split[key]) for key in ("train", "validation", "test")] == [210, 45, 45]
    assert split["train"].timestamp.max() < split["validation"].timestamp.min()
    assert split["validation"].timestamp.max() < split["test"].timestamp.min()
    family = config["dataset"]["heldout_family"]
    assert family not in set(split["train"].family) | set(split["validation"].family)
    assert family in set(split["test"].family)
    known_techniques = pd.concat([split["train"], split["validation"]]).mitre_technique.dropna()
    assert not known_techniques.str.startswith("T1003").any()


def test_generator_determinism(config, alerts):
    pd.testing.assert_frame_equal(alerts, generate_alerts(config, 300))


def test_timestamp_ties_not_split():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10).repeat(3)})
    split = chronological_split(frame, .65, .2)
    assert split["train"].timestamp.max() < split["validation"].timestamp.min()
    assert split["validation"].timestamp.max() < split["test"].timestamp.min()


def test_bad_chronology_fails(alerts):
    with pytest.raises(ValueError, match="valid timestamps"):
        chronological_split(alerts.assign(timestamp="invalid"))


def test_tsv_ingestion_missing_optional_fields(tmp_path):
    path = tmp_path / "alerts.tsv"
    pd.DataFrame({"alert_name": ["a", "b"], "verdict": [" benign ", None]}).to_csv(path, sep="\t", index=False)
    frame = read_alerts(path)
    assert frame.alert_id.is_unique
    assert frame.verdict.iloc[0] == "BENIGN"


def test_duplicate_alert_ids_rejected(tmp_path):
    path = tmp_path / "alerts.csv"
    pd.DataFrame({"alert_id": ["a", "a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_alerts(path)
