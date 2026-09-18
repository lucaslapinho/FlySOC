"""Sparse random expansion and deterministic top-k winner selection."""

import numpy as np
from scipy import sparse


class FlyHash:
    """Mushroom Body-inspired computational motif, not a neural simulation.

    Each KC receives `fan_in` distinct, nonnegative PN connections. Ties are
    resolved by a fixed seeded KC permutation. Only positive activations win:
    empty input never receives an artificial fingerprint.
    """

    def __init__(self, kc_dim: int = 8192, fan_in: int = 8, top_k: int = 64,
                 random_seed: int = 42, batch_size: int = 128):
        self.kc_dim = kc_dim
        self.fan_in = fan_in
        self.top_k = top_k
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.backend_name = "flyhash"

    def fit(self, x: sparse.spmatrix) -> "FlyHash":
        self.input_dim = x.shape[1]
        if not 1 <= self.fan_in <= self.input_dim:
            raise ValueError("fan_in must be within input dimensionality")
        if not 1 <= self.top_k <= self.kc_dim or self.batch_size < 1:
            raise ValueError("Invalid top_k, kc_dim or batch_size")
        rng = np.random.default_rng(self.random_seed)
        rows = np.concatenate([rng.choice(self.input_dim, self.fan_in, replace=False)
                               for _ in range(self.kc_dim)])
        cols = np.repeat(np.arange(self.kc_dim), self.fan_in)
        self.projection_ = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                              shape=(self.input_dim, self.kc_dim))
        self.tie_order_ = rng.permutation(self.kc_dim)
        return self

    def transform(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        if not hasattr(self, "projection_"):
            raise RuntimeError("Fit FlyHash before encoding")
        x = sparse.csr_matrix(x, dtype=np.float32)
        if x.shape[1] != self.input_dim:
            raise ValueError("Unexpected PN dimensionality")
        if not np.isfinite(x.data).all() or np.any(x.data < 0):
            raise ValueError("FlyHash requires finite nonnegative PN features")
        blocks = []
        for start in range(0, x.shape[0], self.batch_size):
            # The only dense KC allocation is batch_size x kc_dim.
            activation = self.transform_activations(x[start:start + self.batch_size]).toarray()
            reordered = activation[:, self.tie_order_]
            order = np.argsort(-reordered, axis=1, kind="stable")[:, :self.top_k]
            columns = self.tie_order_[order]
            values = np.take_along_axis(activation, columns, axis=1)
            rows = np.broadcast_to(np.arange(len(activation))[:, None], columns.shape)
            positive = values > 0
            blocks.append(sparse.csr_matrix((np.ones(positive.sum(), dtype=np.uint8),
                                             (rows[positive], columns[positive])),
                                            shape=(len(activation), self.kc_dim)))
        return sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, self.kc_dim), dtype=np.uint8)

    def transform_activations(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        """Return nonnegative KC activations before winner selection."""
        if not hasattr(self, "projection_"):
            raise RuntimeError("Fit FlyHash before encoding")
        x = sparse.csr_matrix(x, dtype=np.float32)
        if x.shape[1] != self.input_dim:
            raise ValueError("Unexpected PN dimensionality")
        if not np.isfinite(x.data).all() or np.any(x.data < 0):
            raise ValueError("FlyHash requires finite nonnegative PN features")
        return (x @ self.projection_).tocsr()

    def connectivity_matrix(self) -> sparse.csr_matrix:
        """Expose engineered PN-to-KC edges for storage metrics and visualization."""
        if not hasattr(self, "projection_"):
            raise RuntimeError("Fit FlyHash before accessing connectivity")
        return self.projection_

    def fit_transform(self, x: sparse.spmatrix) -> sparse.csr_matrix:
        return self.fit(x).transform(x)
