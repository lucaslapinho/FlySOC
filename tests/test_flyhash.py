import numpy as np
import pytest
from scipy import sparse

from flysoc.flyhash import FlyHash
from flysoc.similarity import binary_similarity


def input_matrix():
    return sparse.csr_matrix(np.random.default_rng(7).random((20, 128)).astype(np.float32))


def test_dimension_topk_and_determinism():
    x = input_matrix()
    a = FlyHash(256, 8, 16, random_seed=42).fit_transform(x)
    b = FlyHash(256, 8, 16, random_seed=42).fit_transform(x)
    assert a.shape == (20, 256)
    assert np.all(a.getnnz(axis=1) == 16)
    assert set(a.data) == {1}
    assert (a != b).nnz == 0


def test_fanin_unique_and_seed_changes_projection():
    x = input_matrix()
    a, b = FlyHash(256, 8, 16, 1).fit(x), FlyHash(256, 8, 16, 2).fit(x)
    assert np.all(a.projection_.getnnz(axis=0) == 8)
    assert (a.projection_ != b.projection_).nnz > 0


def test_ties_reproducible_across_batches():
    x = sparse.csr_matrix(np.ones((9, 20)))
    a = FlyHash(40, 4, 5, batch_size=2).fit_transform(x)
    b = FlyHash(40, 4, 5, batch_size=9).fit_transform(x)
    assert (a != b).nnz == 0
    assert np.all(a.getnnz(axis=1) == 5)


def test_zero_input_never_gets_artificial_winners():
    x = sparse.csr_matrix((4, 128))
    assert FlyHash(256, 8, 16).fit_transform(x).nnz == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_pn_rejected(value):
    x = sparse.csr_matrix([[value, 1]])
    with pytest.raises(ValueError, match="finite nonnegative"):
        FlyHash(8, 2, 3).fit_transform(x)


def test_width_validation():
    encoder = FlyHash(8, 2, 3).fit(sparse.csr_matrix([[1, 1]]))
    with pytest.raises(ValueError, match="dimensionality"):
        encoder.transform(sparse.csr_matrix([[1, 1, 1]]))


def test_jaccard_hamming_and_cosine():
    a, b = sparse.csr_matrix([[1, 1, 0, 0]]), sparse.csr_matrix([[1, 0, 1, 0]])
    assert binary_similarity(a, b)[0, 0] == pytest.approx(1 / 3)
    assert binary_similarity(a, b, "cosine")[0, 0] == pytest.approx(.5)
    assert binary_similarity(a, b, "hamming")[0, 0] == pytest.approx(.5)
    assert binary_similarity(a, a)[0, 0] == 1


def test_binary_overlap_does_not_overflow_uint8():
    x = sparse.csr_matrix(np.ones((1, 300), dtype=np.uint8))
    assert binary_similarity(x, x)[0, 0] == 1
    with pytest.raises(ValueError, match="binary"):
        binary_similarity(x * 2, x)
