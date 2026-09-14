"""Conservative, deterministic exemplar-based deduplication experiment."""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .similarity import binary_similarity


def deduplicate(x: sparse.spmatrix, threshold: float = .75,
                metric: str = "jaccard") -> np.ndarray:
    """Assign each row to its closest previous exemplar above threshold.

    This is order-dependent and does not guarantee pairwise similarity within a
    cluster. It avoids unbounded transitive connected-component chaining.
    """
    if not 0 < threshold <= 1 or metric not in {"jaccard", "cosine"}:
        raise ValueError("Invalid threshold or metric")
    x = normalize(x).tocsr() if metric == "cosine" else sparse.csr_matrix(x)
    representatives: list[int] = []
    assignments = np.empty(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        if representatives:
            if metric == "cosine":
                similarities = (x[i] @ x[representatives].T).toarray().ravel()
            else:
                similarities = binary_similarity(x[i], x[representatives]).ravel()
            closest = int(np.argmax(similarities))
            if similarities[closest] >= threshold:
                assignments[i] = closest
                continue
        assignments[i] = len(representatives)
        representatives.append(i)
    return assignments
