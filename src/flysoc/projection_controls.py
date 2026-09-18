"""Matched topology controls for connectome projection experiments."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def degree_signature(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Return unweighted row and column degrees after canonicalization."""
    canonical = sparse.csr_matrix(matrix, copy=True)
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    return canonical.getnnz(axis=1), canonical.getnnz(axis=0)


def degree_preserving_rewire(
    matrix: sparse.spmatrix,
    swaps_per_edge: float = 5,
    random_seed: int = 42,
) -> sparse.csr_matrix:
    """Randomize a bipartite projection with degree-preserving double-edge swaps.

    A valid swap replaces edges ``(a, b)`` and ``(c, d)`` with ``(a, d)`` and
    ``(c, b)``.  It is accepted only when the two new edges do not already
    exist.  This preserves every row and column degree exactly.  Edge weights
    remain attached to their edge slots, preserving the global weight
    multiset, although weighted row and column strengths are not constrained.

    The returned CSR matrix has a ``rewire_stats_`` diagnostic attribute.  A
    constrained graph may accept fewer than the requested number of swaps;
    callers can use that diagnostic to reject an insufficiently mixed control.
    """
    if not np.isfinite(swaps_per_edge) or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be finite and nonnegative")

    canonical = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    if canonical.nnz < 2:
        raise ValueError("Degree-preserving rewiring requires at least two edges")
    if not np.isfinite(canonical.data).all() or np.any(canonical.data < 0):
        raise ValueError("Projection weights must be finite and nonnegative")

    original_signature = degree_signature(canonical)
    edges = canonical.tocoo()
    rows = edges.row.astype(np.int64, copy=True)
    cols = edges.col.astype(np.int64, copy=True)
    weights = edges.data.copy()
    occupied = set(zip(rows.tolist(), cols.tolist()))

    requested = int(round(float(swaps_per_edge) * canonical.nnz))
    max_attempts = max(1_000, requested * 30)
    accepted = 0
    attempts = 0
    rng = np.random.default_rng(random_seed)

    while accepted < requested and attempts < max_attempts:
        attempts += 1
        first, second = rng.integers(0, canonical.nnz, size=2)
        if first == second:
            continue

        a, b = int(rows[first]), int(cols[first])
        c, d = int(rows[second]), int(cols[second])
        if a == c or b == d:
            continue

        replacement_one = (a, d)
        replacement_two = (c, b)
        if replacement_one in occupied or replacement_two in occupied:
            continue

        occupied.remove((a, b))
        occupied.remove((c, d))
        occupied.add(replacement_one)
        occupied.add(replacement_two)
        cols[first], cols[second] = d, b
        accepted += 1

    rewired = sparse.csr_matrix((weights, (rows, cols)), shape=canonical.shape)
    rewired.sum_duplicates()
    rewired.eliminate_zeros()
    rewired_signature = degree_signature(rewired)
    if not all(
        np.array_equal(before, after)
        for before, after in zip(original_signature, rewired_signature)
    ):
        raise RuntimeError("Internal error: rewiring changed the degree signature")

    rewired.rewire_stats_ = {
        "requested_swaps": requested,
        "accepted_swaps": accepted,
        "attempts": attempts,
        "acceptance_rate": accepted / attempts if attempts else 0.0,
        "completed": accepted == requested,
    }
    return rewired
