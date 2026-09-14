import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flysoc.config import validate_config
from flysoc.preprocessing import AlertPreprocessor


def test_sparse_fixed_dimensions_and_missing_fields(config, alerts):
    pre = AlertPreprocessor(config["features"]).fit(alerts.iloc[:200])
    output = pre.transform(pd.DataFrame({"alert_name": ["new alert", None], "source_port": ["bad", np.inf]}))
    assert sparse.isspmatrix_csr(output)
    assert output.shape == (2, 128)
    assert np.isfinite(output.data).all()
    assert np.all(output.data >= 0)


def test_no_ground_truth_features(config, alerts):
    pre = AlertPreprocessor(config["features"])
    original = pre.fit_transform(alerts)
    altered = alerts.assign(verdict="LEAK", family="LEAK", is_unseen="LEAK", alert_id="LEAK",
                            rule_id="LEAK", mitre_technique="LEAK", mitre_tactic="LEAK")
    assert (original != pre.transform(altered)).nnz == 0


def test_vocabulary_is_fit_on_training_only(config):
    pre = AlertPreprocessor(config["features"]).fit(pd.DataFrame({"alert_name": ["old inventory"]}))
    vocab = pre.vectorizer.vocabulary_.copy()
    pre.transform(pd.DataFrame({"alert_name": ["neverobservedtoken"]}))
    assert pre.vectorizer.vocabulary_ == vocab
    assert "neverobservedtoken" not in vocab


def test_empty_text_and_missing_everything(config):
    pre = AlertPreprocessor(config["features"])
    x = pre.fit_transform(pd.DataFrame(index=range(3)))
    assert x.shape == (3, 128) and x.nnz == 0
    assert pre.transform(pd.DataFrame()).shape == (0, 128)


def test_categorical_high_cardinality_has_bounded_width(config):
    frame = pd.DataFrame({"hostname": [f"host-{i}" for i in range(1000)]})
    x = AlertPreprocessor(config["features"]).fit_transform(frame)
    assert x.shape == (1000, 128)
    assert x.nnz == 1000


def test_preprocessor_requires_fit(config):
    with pytest.raises(RuntimeError):
        AlertPreprocessor(config["features"]).transform(pd.DataFrame({"alert_name": ["a"]}))


def test_configuration_rejects_leakage_and_bad_dimensions(config):
    config["features"]["text_fields"].append("verdict")
    with pytest.raises(ValueError, match="Label"):
        validate_config(config)
    config["features"]["text_fields"].pop()
    config["features"]["input_features"] += 1
    with pytest.raises(ValueError, match="input_features"):
        validate_config(config)
