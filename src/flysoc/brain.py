"""Live numerical traces for FlySOC plus real-connectome exploration."""

import json
from pathlib import Path
import threading
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import json_value
from .connectome import CONNECTOME, ConnectomePulse
from .experiment import load_experiment


def schematic_positions(count: int, kind: str, seed: int = 42) -> np.ndarray:
    """Deterministic display-only positions: no anatomical coordinates implied."""
    rng = np.random.default_rng(seed)
    side = np.where(np.arange(count) % 2, 1, -1)
    angle = rng.uniform(0, 2 * np.pi, count)
    radius = np.sqrt(rng.uniform(0, 1, count))
    if kind == "pn":
        x = side * (53 + 26 * radius * np.cos(angle))
        y = -38 + 20 * radius * np.sin(angle)
        z = rng.normal(0, 10, count)
    else:
        x = side * (39 + 35 * radius * np.cos(angle))
        y = 19 + 27 * radius * np.sin(angle)
        z = rng.normal(0, 12, count)
    return np.column_stack([x, y, z]).astype(np.float32)


class BrainLab:
    def __init__(self, identifier: str | None = None, connectome_path: Path = CONNECTOME):
        self.run_path, self.payload, self.splits = load_experiment(identifier)
        self.pipeline = self.payload["pipeline"]
        self.lock = threading.Lock()
        self.connectome_path = connectome_path
        self.test = self.splits["test"]
        self.pn_positions = schematic_positions(self.pipeline.encoder.input_dim, "pn")
        self.kc_positions = schematic_positions(self.pipeline.encoder.kc_dim, "kc", 43)
        self.connectome_summary = None
        prepared = connectome_path / "prepared"
        if (prepared / "summary.json").exists():
            self.connectome_summary = json.loads((prepared / "summary.json").read_text())
            self.nodes = pd.read_csv(prepared / "nodes.csv.gz", dtype={"root_id": "string"})
            self.pulse = ConnectomePulse(sparse.load_npz(prepared / "transition.npz"))
            self.pulse_source = "ALPN"

    def status(self) -> dict[str, Any]:
        return {"run": self.run_path.name, "pn_dim": self.pipeline.encoder.input_dim,
                "kc_dim": self.pipeline.encoder.kc_dim, "top_k": self.pipeline.encoder.top_k,
                "fan_in": self.pipeline.encoder.fan_in, "training_alerts": len(self.splits["train"]),
                "test_alerts": len(self.test), "novelty_threshold": self.pipeline.novelty.threshold_,
                "connectome": self.connectome_summary,
                "stimulus_classes": [c for c in ["ALPN", "Kenyon_Cell", "MBON", "DAN", "visual", "olfactory"]
                                     if self.connectome_summary and c in self.connectome_summary["class_counts"]]}

    def fly_layout(self) -> dict[str, Any]:
        projection = self.pipeline.encoder.projection_.tocoo()
        sample = np.random.default_rng(42).choice(len(projection.data), min(2500, len(projection.data)), replace=False)
        return {"pn_positions": np.round(self.pn_positions, 3).ravel().tolist(),
                "kc_positions": np.round(self.kc_positions, 3).ravel().tolist(),
                "edges": np.column_stack([projection.row[sample], projection.col[sample]]).ravel().tolist(),
                "anatomical": False, "coordinates": "schematic display-only layout"}

    def alert_catalog(self) -> list[dict[str, Any]]:
        return json_value([{"index": i, "alert_id": row.alert_id, "alert_name": row.get("alert_name", ""),
                            "family": row.get("family", ""), "verdict": row.get("verdict", "UNKNOWN"),
                            "unseen": row.get("family") not in set(self.splits["train"].get("family", []))}
                           for i, (_, row) in enumerate(self.test.iterrows())])

    def trace_alert(self, index: int = 0, alert: dict[str, Any] | None = None) -> dict[str, Any]:
        if alert is None:
            if not isinstance(index, int) or not 0 <= index < len(self.test):
                raise ValueError("Alert index out of range")
            frame = self.test.iloc[[index]].copy()
        else:
            if not isinstance(alert, dict) or not alert or any(isinstance(v, (dict, list)) for v in alert.values()):
                raise ValueError("Provide a flat JSON alert object")
            frame = pd.DataFrame([{**alert, "alert_id": "MANUAL"}])
        started = perf_counter()
        pn = self.pipeline.preprocessor.transform(frame)
        pn_ms = (perf_counter() - started) * 1000
        projected = perf_counter()
        activations = (pn @ self.pipeline.encoder.projection_).toarray()[0]
        fp = self.pipeline.encoder.transform(pn)
        encoding_ms = (perf_counter() - projected) * 1000
        memory_started = perf_counter()
        history = self.pipeline.memory.retrieve(fp, 5)[0]
        novelty = self.pipeline.novelty.score(fp).iloc[0].to_dict()
        winners = fp.indices
        positive_pn = pn.indices
        connections = self.pipeline.encoder.projection_[:, winners].tocoo()
        contributing = np.isin(connections.row, positive_pn)
        edges = np.column_stack([connections.row[contributing], winners[connections.col[contributing]]])
        prediction = None
        model = self.payload["classifiers"].get("fly_logistic_regression")
        if model is not None:
            probabilities = model.predict_proba(fp)[0]
            prediction = {"model": "Fly Logistic Regression", "verdict": str(model.classes_[probabilities.argmax()]),
                          "probabilities": dict(zip(model.classes_, probabilities.tolist()))}
        return json_value({"source": "live Python calculation", "alert_index": index if alert is None else None,
                           "alert": frame.iloc[0].to_dict(), "pn_indices": positive_pn.tolist(), "pn_values": pn.data.tolist(),
                           "kc_activations": activations.tolist(), "winner_indices": winners.tolist(),
                           "contributing_edges": edges.ravel().tolist(), "active_kcs": len(winners),
                           "nonzero_activations": int(np.count_nonzero(activations)),
                           "history": [{k: row.get(k) for k in ["alert_id", "alert_name", "verdict", "similarity", "hostname"]} for row in history],
                           "novelty": novelty, "novelty_threshold": self.pipeline.novelty.threshold_, "prediction": prediction,
                           "timing_ms": {"pn": pn_ms, "encoding": encoding_ms,
                                          "memory": (perf_counter() - memory_started) * 1000,
                                          "total": (perf_counter() - started) * 1000}})

    def pulse_frame(self, reset_class: str | None = None, advance: bool = True) -> dict[str, Any]:
        if self.connectome_summary is None:
            raise ValueError("Download and prepare the connectome first")
        with self.lock:
            if reset_class is not None:
                indices = np.flatnonzero((self.nodes["class"] == reset_class).to_numpy())
                if not len(indices):
                    raise ValueError("Unknown stimulation class")
                self.pulse.reset(indices)
                self.pulse_source = reset_class
            start = perf_counter()
            activity = self.pulse.step() if advance else self.pulse.activity.copy()
            elapsed = (perf_counter() - start) * 1000
            positive = np.flatnonzero(activity > 1e-8)
            top = positive[np.argsort(activity[positive])[-2000:][::-1]]
            leaders = top[:8]
            return {"iteration": self.pulse.iteration, "source_class": self.pulse_source,
                    "source": "live unsigned graph diffusion", "compute_ms": elapsed,
                    "active_neurons": len(positive), "activity_mass": float(activity.sum()),
                    "max_activity": float(activity.max()), "node_indices": top.tolist(),
                    "node_values": activity[top].tolist(), "display_limit": 2000,
                    "leaders": [{"root_id": str(self.nodes.root_id.iloc[i]), "class": str(self.nodes["class"].iloc[i]),
                                 "activity": float(activity[i])} for i in leaders]}
