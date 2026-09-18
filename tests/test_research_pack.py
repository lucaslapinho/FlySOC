"""Focused contracts for optional research architecture."""

import hashlib
import json

import numpy as np
from scipy import sparse

from flysoc.config import load_config
from flysoc.connectome_projection import ConnectomeHash
from flysoc.mbon_learning import MBONInspiredReadout
from flysoc.pipeline import FlySOCPipeline
from flysoc.projection_controls import degree_preserving_rewire, degree_signature


def test_connectome_hash_is_deterministic_and_sparse():
    biological = sparse.csr_matrix(np.array([
        [1, 0, 2, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0],
    ], dtype=np.float32))
    x = sparse.csr_matrix(np.array([[1, 0, 2, 0], [0, 1, 0, 3]], dtype=np.float32))
    first = ConnectomeHash(biological, input_dim=4, top_k=2, bridge_fan_out=1, random_seed=7)
    second = ConnectomeHash(biological, input_dim=4, top_k=2, bridge_fan_out=1, random_seed=7)
    first_fp, second_fp = first.transform(x), second.transform(x)
    assert (first_fp != second_fp).nnz == 0
    assert first_fp.shape == (2, 5)
    assert np.all(first_fp.getnnz(axis=1) <= 2)


def test_pipeline_selects_verified_connectome_backends(tmp_path):
    projection = sparse.csr_matrix(np.array([
        [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1],
    ], dtype=np.float32))
    artifact = tmp_path / "pn_to_kc.npz"
    sparse.save_npz(artifact, projection)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "projection": {"shape": list(projection.shape), "nnz": projection.nnz},
        "outputs": {"pn_to_kc.npz": {"sha256": digest}},
    }), encoding="utf-8")

    config = load_config()
    config["connectome"].update({
        "enabled": True,
        "projection_path": str(tmp_path),
        "bridge_fan_out": 1,
        "top_k": 2,
    })
    config["representation"]["backend"] = "connectome"
    assert FlySOCPipeline(config).encoder.backend_name == "connectome"

    config["representation"]["backend"] = "rewired_connectome"
    config["controls"]["rewired"]["enabled"] = True
    assert FlySOCPipeline(config).encoder.backend_name == "rewired_connectome"


def test_rewire_preserves_bipartite_degree():
    matrix = sparse.csr_matrix(np.array([
        [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1],
    ], dtype=np.float32))
    rewired = degree_preserving_rewire(matrix, swaps_per_edge=2, random_seed=3)
    before, after = degree_signature(matrix), degree_signature(rewired)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_mbon_explicit_feedback_moves_target_probability():
    fingerprint = sparse.csr_matrix([[1, 0, 1, 0]], dtype=np.float32)
    model = MBONInspiredReadout(
        4,
        classes=("TP", "FP", "BENIGN"),
        reward_policy={"TP": 1.0, "FP": -0.5, "BENIGN": 0.0},
        learning_rate=0.5,
    )
    before = model.predict_proba(fingerprint)[0, 0]
    for _ in range(10):
        model.partial_fit(fingerprint, ["TP"])
    assert model.predict_proba(fingerprint)[0, 0] > before
    assert len(model.history_) == 10
