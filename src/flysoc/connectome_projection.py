"""Optional connectome-constrained sparse projection for FlySOC.

The measured artifact constrains only the biological PN-to-KC stage.  Mapping
engineered alert features into PN coordinates is an explicit deterministic
bridge; it is not measured sensory physiology.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


ARTIFACT_NAME = "pn_to_kc.npz"
MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_csr(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise TypeError("PN-to-KC projection must be a scipy sparse matrix")
    result = sparse.csr_matrix(matrix, dtype=np.float32)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("PN-to-KC projection must be non-empty")
    if not np.isfinite(result.data).all() or np.any(result.data <= 0):
        raise ValueError("PN-to-KC weights must be finite and strictly positive")
    return result


def load_connectome_projection(
    artifact: str | Path | None,
    *,
    required: bool = True,
    verify_sha256: bool = True,
) -> tuple[sparse.csr_matrix, dict[str, Any]] | None:
    """Load and validate a prepared PN-to-KC CSR artifact and its provenance.

    ``artifact`` may be the build output directory or its ``pn_to_kc.npz``.
    When connectome mode is optional, callers can pass ``required=False`` and
    treat a missing path as an explicit signal to keep using :class:`FlyHash`.
    Present but corrupt or inconsistent artifacts always raise; they never
    silently fall back to a synthetic projection.
    """
    if artifact is None:
        if required:
            raise FileNotFoundError(
                "Connectome projection was requested but no artifact path was configured"
            )
        return None

    supplied = Path(artifact).expanduser()
    matrix_path = supplied / ARTIFACT_NAME if supplied.is_dir() else supplied
    manifest_path = matrix_path.parent / MANIFEST_NAME
    if not matrix_path.is_file() or not manifest_path.is_file():
        if not required and not supplied.exists():
            return None
        missing = [str(path) for path in (matrix_path, manifest_path) if not path.is_file()]
        raise FileNotFoundError(
            "Incomplete connectome projection artifact; missing " + ", ".join(missing)
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid connectome provenance manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Connectome provenance manifest must contain a JSON object")

    try:
        matrix = _validated_csr(sparse.load_npz(matrix_path))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid connectome CSR artifact: {matrix_path}") from exc

    recorded_shape = manifest.get("projection", {}).get("shape", manifest.get("shape"))
    recorded_nnz = manifest.get("projection", {}).get("nnz", manifest.get("pn_to_kc_edges"))
    if recorded_shape is None or list(recorded_shape) != list(matrix.shape):
        raise ValueError("Connectome CSR shape does not match its provenance manifest")
    if recorded_nnz is None or int(recorded_nnz) != matrix.nnz:
        raise ValueError("Connectome CSR nnz does not match its provenance manifest")

    if verify_sha256:
        recorded_sha = manifest.get("outputs", {}).get(ARTIFACT_NAME, {}).get("sha256")
        if not recorded_sha:
            raise ValueError("Connectome provenance manifest has no projection SHA-256")
        actual_sha = _sha256(matrix_path)
        if actual_sha.lower() != str(recorded_sha).lower():
            raise ValueError("Connectome projection SHA-256 does not match its manifest")

    return matrix, manifest


class ConnectomeHash:
    """Encode engineered FlySOC features through a measured PN-to-KC topology.

    Parameters
    ----------
    pn_to_kc:
        Sparse, finite, positive matrix shaped ``(biological_pn, biological_kc)``.
    input_dim:
        Width of the engineered FlySOC feature vector.
    bridge_fan_out:
        Biological PN coordinates assigned to each engineered input feature.
        This seeded bridge is an engineering adapter, not biological evidence.
    provenance:
        Manifest returned by :func:`load_connectome_projection`.
    """

    backend_name = "connectome"

    def __init__(
        self,
        pn_to_kc: sparse.spmatrix,
        input_dim: int,
        top_k: int = 64,
        bridge_fan_out: int = 2,
        random_seed: int = 42,
        batch_size: int = 128,
        provenance: dict[str, Any] | None = None,
    ):
        matrix = _validated_csr(pn_to_kc)
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not 1 <= bridge_fan_out <= matrix.shape[0]:
            raise ValueError("bridge_fan_out must be within biological PN dimensionality")
        if not 1 <= top_k <= matrix.shape[1]:
            raise ValueError("top_k must be within biological KC dimensionality")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.pn_to_kc = matrix
        self.input_dim = int(input_dim)
        self.top_k = int(top_k)
        self.bridge_fan_out = int(bridge_fan_out)
        self.random_seed = int(random_seed)
        self.batch_size = int(batch_size)
        self.provenance = dict(provenance or {})
        self._build_bridge()

    @classmethod
    def from_artifact(
        cls,
        artifact: str | Path,
        input_dim: int,
        **kwargs: Any,
    ) -> "ConnectomeHash":
        """Create an encoder from a verified on-disk research artifact."""
        loaded = load_connectome_projection(artifact, required=True)
        if loaded is None:  # pragma: no cover - required=True makes this impossible
            raise RuntimeError("Connectome artifact unexpectedly unavailable")
        matrix, manifest = loaded
        return cls(matrix, input_dim=input_dim, provenance=manifest, **kwargs)

    @property
    def kc_dim(self) -> int:
        return self.pn_to_kc.shape[1]

    @property
    def biological_pn_dim(self) -> int:
        return self.pn_to_kc.shape[0]

    def _build_bridge(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        rows: list[int] = []
        columns: list[int] = []
        for feature in range(self.input_dim):
            targets = rng.choice(
                self.biological_pn_dim, self.bridge_fan_out, replace=False
            )
            rows.extend([feature] * self.bridge_fan_out)
            columns.extend(targets.tolist())
        weights = np.full(
            len(rows), 1.0 / self.bridge_fan_out, dtype=np.float32
        )
        self.bridge_ = sparse.csr_matrix(
            (weights, (rows, columns)),
            shape=(self.input_dim, self.biological_pn_dim),
        )
        self.tie_order_ = rng.permutation(self.kc_dim)

    def connectivity_matrix(self) -> sparse.csr_matrix:
        """Return effective engineered-feature-to-KC connectivity.

        The product is computed on demand because its number of nonzeros can be
        much larger than either the bridge or measured PN-to-KC input matrix.
        """
        effective = (self.bridge_ @ self.pn_to_kc).tocsr()
        effective.sum_duplicates()
        effective.eliminate_zeros()
        effective.sort_indices()
        return effective

    def fit(self, x: sparse.spmatrix) -> "ConnectomeHash":
        if x.shape[1] != self.input_dim:
            raise ValueError("Unexpected engineered feature dimensionality")
        return self

    def transform_activations(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        x = sparse.csr_matrix(x, dtype=np.float32)
        if x.shape[1] != self.input_dim:
            raise ValueError("Unexpected engineered feature dimensionality")
        if not np.isfinite(x.data).all() or np.any(x.data < 0):
            raise ValueError("ConnectomeHash requires finite nonnegative inputs")
        biological_pn = normalize(x @ self.bridge_, norm="l2").tocsr()
        activation = (biological_pn @ self.pn_to_kc).tocsr()
        activation.eliminate_zeros()
        return activation

    def transform(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        activation = self.transform_activations(x)
        blocks: list[sparse.csr_matrix] = []
        for start in range(0, activation.shape[0], self.batch_size):
            dense = activation[start : start + self.batch_size].toarray()
            reordered = dense[:, self.tie_order_]
            order = np.argsort(-reordered, axis=1, kind="stable")[:, : self.top_k]
            columns = self.tie_order_[order]
            values = np.take_along_axis(dense, columns, axis=1)
            rows = np.broadcast_to(np.arange(len(dense))[:, None], columns.shape)
            positive = values > 0
            blocks.append(
                sparse.csr_matrix(
                    (
                        np.ones(int(positive.sum()), dtype=np.uint8),
                        (rows[positive], columns[positive]),
                    ),
                    shape=(len(dense), self.kc_dim),
                )
            )
        if not blocks:
            return sparse.csr_matrix((0, self.kc_dim), dtype=np.uint8)
        return sparse.vstack(blocks, format="csr")

    def fit_transform(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        return self.fit(x).transform(x)
