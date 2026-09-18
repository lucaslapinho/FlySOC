"""Build a measured MaleCNS PN-to-KC CSR projection from local Feather files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from scipy import sparse


ARTIFACT_NAME = "pn_to_kc.npz"
MANIFEST_NAME = "manifest.json"
PN_CANDIDATES_NAME = "pn_candidates.csv"
KC_CANDIDATES_NAME = "kc_candidates.csv"

ANNOTATION_ID_ALIASES = (
    "bodyId", "body_id", "body", "root_id", "rootId", "neuron_id", "id"
)
PRE_ID_ALIASES = (
    "pre_body_id", "body_pre", "pre_body", "pre_bodyId", "bodyId_pre",
    "pre_root_id", "pre_rootId", "presynaptic_body_id", "pre"
)
POST_ID_ALIASES = (
    "post_body_id", "body_post", "post_body", "post_bodyId", "bodyId_post",
    "post_root_id", "post_rootId", "postsynaptic_body_id", "post"
)
WEIGHT_ALIASES = (
    "synapse_count", "syn_count", "synapses", "weight", "count", "n"
)
ANNOTATION_TEXT_ALIASES = {
    "type", "instance", "class", "super_class", "superclass", "sub_class",
    "subclass", "cell_type", "celltype", "name", "hemibrain_type",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def choose_column(columns: Iterable[object], aliases: Iterable[str], role: str) -> object:
    """Resolve a known MaleCNS alias while rejecting ambiguous schemas."""
    normalized: dict[str, list[object]] = {}
    for column in columns:
        normalized.setdefault(_normalized_column(column), []).append(column)
    for alias in aliases:
        matches = normalized.get(_normalized_column(alias), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {role} columns for alias {alias!r}: {matches}")
    raise ValueError(
        f"Could not detect {role}; accepted aliases={list(aliases)}, "
        f"available columns={list(columns)}"
    )


def annotation_text(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    selected = [
        column for column in frame.columns
        if _normalized_column(column) in {
            _normalized_column(alias) for alias in ANNOTATION_TEXT_ALIASES
        }
    ]
    if not selected:
        selected = [
            column for column in frame.columns
            if is_string_dtype(frame[column].dtype) or is_object_dtype(frame[column].dtype)
        ][:8]
    if not selected:
        raise ValueError("No textual annotation columns are available for PN/KC selection")
    text = pd.Series("", index=frame.index, dtype="string")
    for column in selected:
        text = text + " " + frame[column].astype("string").fillna("")
    return text.str.strip(), [str(column) for column in selected]


def canonical_ids(series: pd.Series, role: str) -> pd.Series:
    result = series.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{role} contains missing neuron IDs")
    return result


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "modified_at_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local MaleCNS PN-to-KC projection; no data is downloaded."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument(
        "--pn-regex",
        default=r"(?i)(?:^|[^a-z])(?:alpn|upn|mpn|pn)(?:[^a-z]|$)|projection neuron",
    )
    parser.add_argument(
        "--kc-regex", default=r"(?i)(?:^|[^a-z])kc(?:[^a-z]|$)|kenyon"
    )
    parser.add_argument("--min-synapses", type=float, default=1.0)
    parser.add_argument(
        "--weight-mode",
        choices=("binary", "synapse", "column_normalized"),
        default="column_normalized",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.min_synapses) or args.min_synapses <= 0:
        raise ValueError("--min-synapses must be finite and positive")
    for source in (args.annotations, args.weights):
        if not source.is_file():
            raise FileNotFoundError(f"Required local MaleCNS source is missing: {source}")

    try:
        pn_pattern = re.compile(args.pn_regex)
        kc_pattern = re.compile(args.kc_regex)
    except re.error as exc:
        raise ValueError(f"Invalid PN/KC selection regex: {exc}") from exc

    annotations = pd.read_feather(args.annotations)
    edges = pd.read_feather(args.weights)
    id_column = choose_column(
        annotations.columns, ANNOTATION_ID_ALIASES, "annotation neuron ID"
    )
    pre_column = choose_column(edges.columns, PRE_ID_ALIASES, "presynaptic neuron ID")
    post_column = choose_column(edges.columns, POST_ID_ALIASES, "postsynaptic neuron ID")
    weight_column = choose_column(edges.columns, WEIGHT_ALIASES, "edge weight")

    text, text_columns = annotation_text(annotations)
    pn_mask = text.str.contains(pn_pattern, na=False)
    kc_mask = text.str.contains(kc_pattern, na=False)
    if (pn_mask & kc_mask).any():
        overlap = int((pn_mask & kc_mask).sum())
        raise ValueError(
            f"PN/KC regexes overlap on {overlap} annotations; refine the selectors"
        )
    pn_candidates = annotations.loc[pn_mask].copy()
    kc_candidates = annotations.loc[kc_mask].copy()
    if pn_candidates.empty or kc_candidates.empty:
        raise ValueError(
            f"Regex selection yielded PN={len(pn_candidates)}, KC={len(kc_candidates)}; "
            "inspect the recorded annotation columns and override the regexes"
        )

    pn_ids = pd.Index(
        canonical_ids(pn_candidates[id_column], "PN annotations").drop_duplicates()
    )
    kc_ids = pd.Index(
        canonical_ids(kc_candidates[id_column], "KC annotations").drop_duplicates()
    )
    if len(pn_ids.intersection(kc_ids)):
        raise ValueError("Selected PN and KC neuron ID sets overlap")

    edge_data = pd.DataFrame({
        "pre": canonical_ids(edges[pre_column], "edge pre IDs"),
        "post": canonical_ids(edges[post_column], "edge post IDs"),
        "weight": pd.to_numeric(edges[weight_column], errors="coerce"),
    })
    finite_weight = np.isfinite(edge_data["weight"].to_numpy(dtype=float))
    invalid_weight_rows = int((~finite_weight).sum())
    eligible = (
        edge_data["pre"].isin(pn_ids)
        & edge_data["post"].isin(kc_ids)
        & finite_weight
        & edge_data["weight"].ge(args.min_synapses)
    )
    filtered = edge_data.loc[eligible].copy()
    if filtered.empty:
        raise ValueError("No PN-to-KC edges matched the selected annotations and threshold")
    filtered = filtered.groupby(["pre", "post"], as_index=False, sort=False)["weight"].sum()
    if not np.isfinite(filtered["weight"]).all() or filtered["weight"].le(0).any():
        raise ValueError("Aggregated PN-to-KC weights must be finite and positive")

    rows = pn_ids.get_indexer(filtered["pre"])
    columns = kc_ids.get_indexer(filtered["post"])
    if (rows < 0).any() or (columns < 0).any():
        raise RuntimeError("Internal PN/KC index construction failed")
    values = filtered["weight"].to_numpy(dtype=np.float32)
    if args.weight_mode == "binary":
        values.fill(1)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(pn_ids), len(kc_ids)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if args.weight_mode == "column_normalized":
        totals = np.asarray(matrix.sum(axis=0)).ravel()
        inverse = np.divide(
            1.0, totals, out=np.zeros_like(totals, dtype=np.float32), where=totals > 0
        )
        matrix = (matrix @ sparse.diags(inverse, dtype=np.float32)).tocsr()
        matrix.eliminate_zeros()

    source_manifest: dict[str, Any] | None = None
    source_manifest_path = args.source_manifest
    if source_manifest_path is None and args.annotations.parent == args.weights.parent:
        candidate = args.annotations.parent / "flysoc_source_manifest.json"
        if candidate.is_file():
            source_manifest_path = candidate
    if source_manifest_path is not None:
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"Source manifest is missing: {source_manifest_path}")
        source_manifest = {
            "file": file_record(source_manifest_path),
            "contents": json.loads(source_manifest_path.read_text(encoding="utf-8")),
        }

    output_files = {
        ARTIFACT_NAME: args.output / ARTIFACT_NAME,
        PN_CANDIDATES_NAME: args.output / PN_CANDIDATES_NAME,
        KC_CANDIDATES_NAME: args.output / KC_CANDIDATES_NAME,
        MANIFEST_NAME: args.output / MANIFEST_NAME,
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing projection outputs: " + ", ".join(existing)
        )
    args.output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_files[ARTIFACT_NAME], matrix)
    pn_candidates.drop_duplicates(id_column).to_csv(
        output_files[PN_CANDIDATES_NAME], index=False
    )
    kc_candidates.drop_duplicates(id_column).to_csv(
        output_files[KC_CANDIDATES_NAME], index=False
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "MaleCNS",
            "version": "v1.0",
            "official_project": "https://male-cns.janelia.org/",
            "interpretation": "Measured MaleCNS connectivity; annotation regex selections require scientific review.",
        },
        "purpose": "FlySOC connectome-constrained PN-to-KC research projection",
        "sources": {
            "annotations": file_record(args.annotations),
            "weights": file_record(args.weights),
            "source_manifest": source_manifest,
        },
        "schema": {
            "annotation_columns": [str(column) for column in annotations.columns],
            "weight_columns": [str(column) for column in edges.columns],
            "detected": {
                "annotation_id": str(id_column),
                "pre_id": str(pre_column),
                "post_id": str(post_column),
                "weight": str(weight_column),
                "annotation_text": text_columns,
            },
        },
        "selection": {
            "pn_regex": args.pn_regex,
            "kc_regex": args.kc_regex,
            "min_synapses": args.min_synapses,
            "weight_mode": args.weight_mode,
        },
        "counts": {
            "annotation_rows": int(len(annotations)),
            "weight_rows": int(len(edges)),
            "invalid_weight_rows": invalid_weight_rows,
            "pn_annotation_rows": int(len(pn_candidates)),
            "kc_annotation_rows": int(len(kc_candidates)),
            "pn_ids": int(len(pn_ids)),
            "kc_ids": int(len(kc_ids)),
            "matched_unique_edges": int(len(filtered)),
        },
        "projection": {
            "format": "scipy.sparse.csr_matrix",
            "dtype": str(matrix.dtype),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "bridge_note": "The downstream engineered-feature-to-PN bridge is seeded and is not biological data.",
        },
        "outputs": {
            name: file_record(path)
            for name, path in output_files.items()
            if name != MANIFEST_NAME
        },
        "warning": (
            "Regex-based PN/KC selection must be reviewed against official MaleCNS "
            "annotations before scientific interpretation."
        ),
    }
    output_files[MANIFEST_NAME].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
