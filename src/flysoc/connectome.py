"""Real FlyWire graph preparation and explicitly abstract signal propagation."""

import gzip
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import sha256, write_json
from .config import ROOT

CONNECTOME = ROOT / "data/connectome/fafb783"


def parse_positions(series: pd.Series) -> np.ndarray:
    """Read three finite numeric coordinates. Never evaluate source strings."""
    values = series.str.strip("[] ").str.split(expand=True)
    if values.shape[1] != 3:
        raise ValueError("Expected exactly three coordinates per position")
    result = values.to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Coordinates must be finite")
    return result


def prepare_connectome(base: Path = CONNECTOME) -> dict[str, Any]:
    """Build a weighted CSR graph and a bounded visual edge sample.

    All neurons and connection rows participate in the simulation graph.
    Display positions are community-marked coordinates, not soma claims or
    neuron skeletons. Display links are straight chords, not reconstructed axons.
    """
    raw, prepared = base / "raw", base / "prepared"
    manifest = json.loads((base / "download_manifest.json").read_text())
    for name, record in manifest["files"].items():
        if sha256(raw / name) != record["sha256"]:
            raise ValueError(f"Connectome checksum mismatch: {name}")
    if prepared.exists():
        saved = json.loads((prepared / "summary.json").read_text())
        if saved["source_sha256"] != {n: r["sha256"] for n, r in manifest["files"].items()}:
            raise ValueError("Prepared data belongs to a different source snapshot; preserve it separately")
        return saved
    started = perf_counter()
    neurons = pd.read_csv(raw / "neurons.csv.gz", dtype={"root_id": "int64"})
    classification = pd.read_csv(raw / "classification.csv.gz", dtype={"root_id": "int64"})
    coordinates = pd.read_csv(raw / "coordinates.csv.gz", dtype={"root_id": "int64"})
    if neurons.root_id.duplicated().any():
        raise ValueError("Neuron IDs must be unique")
    nodes = neurons[["root_id", "nt_type"]].merge(classification, on="root_id", how="left", validate="one_to_one")
    ids = pd.Index(nodes.root_id)
    coordinate_indices = ids.get_indexer(coordinates.root_id)
    if (coordinate_indices < 0).any():
        raise ValueError("Position references an unknown neuron")
    xyz = parse_positions(coordinates.position)
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    scale = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 230
    visual_xyz = ((xyz - center) / scale).astype(np.float32)
    visual_xyz[:, 1:] *= -1  # explicit display orientation; preserve source in raw
    first = ~coordinates.root_id.duplicated().to_numpy()
    representative = np.zeros((len(nodes), 3), dtype=np.float32)
    representative[coordinate_indices[first]] = visual_xyz[first]
    located = np.zeros(len(nodes), dtype=bool)
    located[coordinate_indices] = True

    row_parts, col_parts, weight_parts = [], [], []
    connection_rows = 0
    synapse_total = 0
    for chunk in pd.read_csv(raw / "connections.csv.gz", chunksize=250000,
                             usecols=["pre_root_id", "post_root_id", "syn_count"],
                             dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"}):
        pre, post = ids.get_indexer(chunk.pre_root_id), ids.get_indexer(chunk.post_root_id)
        weights = chunk.syn_count.to_numpy()
        if (pre < 0).any() or (post < 0).any() or (weights <= 0).any():
            raise ValueError("Invalid connection endpoints or weights")
        row_parts.append(pre.astype(np.int32))
        col_parts.append(post.astype(np.int32))
        weight_parts.append(weights)
        connection_rows += len(chunk)
        synapse_total += int(weights.sum())
    graph = sparse.csr_matrix((np.concatenate(weight_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
                               shape=(len(nodes), len(nodes)), dtype=np.float32)
    graph.sum_duplicates()
    outgoing = np.asarray(graph.sum(axis=1)).ravel()
    transition = sparse.diags(np.divide(1, outgoing, out=np.zeros_like(outgoing), where=outgoing > 0)) @ graph
    # Uniform sample of real aggregated graph edges. It is for visibility only.
    coo = graph.tocoo()
    valid_edges = np.flatnonzero(located[coo.row] & located[coo.col])
    chosen = np.random.default_rng(42).choice(valid_edges, min(18000, len(valid_edges)), replace=False)
    order = chosen[np.argsort(coo.data[chosen], kind="stable")]
    edges = np.column_stack([coo.row[order], coo.col[order]]).astype(np.int32)
    groups = nodes.super_class.fillna("unclassified")
    group_names = sorted(groups.unique())
    group_codes = pd.Categorical(groups, categories=group_names).codes.astype(np.int16)
    node_classes = nodes["class"].fillna("unclassified")
    summary = {"dataset": "FlyWire FAFB", "version": "783", "neurons": len(nodes),
               "connection_rows": connection_rows, "neuron_pairs": graph.nnz,
               "synapses": synapse_total, "coordinate_points": len(xyz), "located_neurons": int(located.sum()),
               "rendered_edges": len(edges), "class_counts": node_classes.value_counts().to_dict(),
               "superclass_counts": groups.value_counts().to_dict(),
               "source_sha256": {n: r["sha256"] for n, r in manifest["files"].items()},
               "coordinate_interpretation": "community-marked positions in nanometers; not soma positions or skeletons",
               "display_transform": {"center_nm": center.tolist(), "scale_nm_per_unit": scale, "axis_signs": [1, -1, -1]},
               "propagation_model": "a[t+1] = 0.90 * (0.20*a[t] + 0.80*P.T @ a[t]); P is row-normalized synapse-count adjacency. Unsigned, discrete, no electrophysiology.",
               "prepare_seconds": perf_counter() - started}
    prepared.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(prepared / "adjacency.npz", graph)
    sparse.save_npz(prepared / "transition.npz", transition.tocsr())
    np.savez_compressed(prepared / "layout.npz", points=visual_xyz, point_nodes=coordinate_indices.astype(np.int32),
                        representative=representative, edges=edges, group_codes=group_codes)
    nodes.to_csv(prepared / "nodes.csv.gz", index=False)
    scene = {"positions": np.round(visual_xyz, 3).ravel().tolist(), "point_nodes": coordinate_indices.tolist(),
             "representative_positions": np.round(representative, 3).ravel().tolist(),
             "edges": edges.ravel().tolist(), "group_codes": group_codes.tolist(), "group_names": group_names,
             "summary": summary}
    with gzip.open(prepared / "scene.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(scene, handle, separators=(",", ":"))
    write_json(prepared / "summary.json", summary)
    return summary


class ConnectomePulse:
    """Stable, unsigned graph diffusion for exploratory tests, not brain activity.

    Nonnegative activity mass cannot increase without external input. Simulation
    steps are abstract iterations; visual playback speed is not biological time.
    """

    def __init__(self, transition: sparse.spmatrix, retention: float = .90, mixing: float = .80):
        if not 0 <= retention <= 1 or not 0 <= mixing <= 1:
            raise ValueError("Retention and mixing must lie in [0, 1]")
        self.transition = sparse.csr_matrix(transition, dtype=np.float32)
        if self.transition.shape[0] != self.transition.shape[1]:
            raise ValueError("Transition matrix must be square")
        if not np.isfinite(self.transition.data).all() or np.any(self.transition.data < 0):
            raise ValueError("Transition weights must be finite and nonnegative")
        if np.max(np.asarray(self.transition.sum(axis=1))) > 1.0001:
            raise ValueError("Transition must be row-substochastic")
        self.retention, self.mixing = retention, mixing
        self.activity = np.zeros(self.transition.shape[0], dtype=np.float32)
        self.iteration = 0

    def reset(self, indices: np.ndarray, amplitude: float = 1) -> None:
        indices = np.asarray(indices, dtype=int)
        if np.any((indices < 0) | (indices >= len(self.activity))) or not np.isfinite(amplitude) or not 0 <= amplitude <= 10:
            raise ValueError("Invalid stimulation")
        self.activity.fill(0)
        self.activity[indices] = amplitude
        self.iteration = 0

    def step(self) -> np.ndarray:
        self.activity = self.retention * ((1 - self.mixing) * self.activity + self.mixing * (self.transition.T @ self.activity))
        self.iteration += 1
        return self.activity.copy()
