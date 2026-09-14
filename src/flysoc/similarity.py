"""Bounded-memory exact retrieval for sparse input and binary fingerprints."""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def binary_matrix(x: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(x, dtype=np.float32, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or not np.isin(matrix.data, [1]).all():
        raise ValueError("Expected a binary fingerprint matrix")
    return matrix


def binary_similarity(a: sparse.spmatrix, b: sparse.spmatrix,
                      metric: str = "jaccard") -> np.ndarray:
    """Pairwise similarity; callers should batch large matrices.

    Hamming similarity = 1 - (symmetric difference / total KC dimensions).
    Zero fingerprints have similarity zero, including to another zero vector.
    """
    a, b = binary_matrix(a), binary_matrix(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Fingerprint widths differ")
    intersection = (a @ b.T).toarray()  # float32 avoids uint8 overlap overflow
    sizes_a, sizes_b = a.getnnz(axis=1)[:, None], b.getnnz(axis=1)[None, :]
    if metric == "jaccard":
        denominator = sizes_a + sizes_b - intersection
        result = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)
    elif metric == "cosine":
        denominator = np.sqrt(sizes_a * sizes_b)
        result = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)
    elif metric == "hamming":
        result = 1 - (sizes_a + sizes_b - 2 * intersection) / a.shape[1]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return np.where((sizes_a > 0) & (sizes_b > 0), result, 0).astype(np.float32)


class SparseIndex:
    """Exact nearest neighbors with deterministic index-order tie resolution."""

    def __init__(self, reference: sparse.spmatrix, metric: str = "cosine", batch_size: int = 128):
        if reference.shape[0] == 0 or batch_size < 1:
            raise ValueError("Index requires reference rows and a positive batch size")
        if metric not in {"cosine", "jaccard", "hamming"}:
            raise ValueError("Unsupported similarity metric")
        self.metric, self.batch_size = metric, batch_size
        self.reference = normalize(sparse.csr_matrix(reference, dtype=np.float32)).tocsr() if metric == "cosine" else binary_matrix(reference)

    def query(self, queries: sparse.spmatrix, k: int = 5,
              exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return descending similarities and reference row indices.

        exclude specifies one reference row per query, used for leave-one-out
        calibration. No full N x N similarity matrix is retained.
        """
        if k < 1 or queries.shape[1] != self.reference.shape[1]:
            raise ValueError("Invalid k or query width")
        if exclude is not None:
            exclude = np.asarray(exclude, dtype=int)
            if exclude.shape != (queries.shape[0],) or np.any((exclude < 0) | (exclude >= self.reference.shape[0])):
                raise ValueError("Invalid self-exclusion indices")
        available = self.reference.shape[0] - (exclude is not None)
        if available < 1:
            raise ValueError("No historical neighbors after self exclusion")
        k = min(k, available)
        q = sparse.csr_matrix(queries, dtype=np.float32)
        if self.metric == "cosine":
            q = normalize(q).tocsr()
        scores, indices = [], []
        for start in range(0, q.shape[0], self.batch_size):
            batch = q[start:start + self.batch_size]
            sim = (batch @ self.reference.T).toarray() if self.metric == "cosine" else binary_similarity(batch, self.reference, self.metric)
            np.clip(sim, 0, 1, out=sim)
            if exclude is not None:
                sim[np.arange(len(sim)), exclude[start:start + len(sim)]] = -np.inf
            order = np.argsort(-sim, axis=1, kind="stable")[:, :k]
            indices.append(order)
            scores.append(np.take_along_axis(sim, order, axis=1))
        if not scores:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=int)
        return np.vstack(scores), np.vstack(indices)


def sparse_bytes(matrix: sparse.spmatrix) -> int:
    matrix = sparse.csr_matrix(matrix)
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
