import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flysoc.memory import AssociativeMemory
from flysoc.novelty import NearestNeighborNovelty
from flysoc.similarity import SparseIndex


def test_topk_history_and_feedback():
    x = sparse.csr_matrix([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])
    metadata = pd.DataFrame({"alert_id": ["a", "b", "c"], "verdict": ["BENIGN"] * 3})
    memory = AssociativeMemory().fit(x, metadata)
    assert memory.retrieve(x[0], 2)[0][0]["alert_id"] == "a"
    assert memory.retrieve(x[0], 2)[0][0]["similarity"] == 1
    memory.update_feedback("a", "FALSE_POSITIVE", "Reviewed recurring inventory")
    assert memory.retrieve(x[0], 1)[0][0]["verdict"] == "FALSE_POSITIVE"
    assert memory.feedback_log[0]["previous_verdict"] == "BENIGN"
    with pytest.raises(ValueError):
        memory.update_feedback("a", "BAD_LABEL")
    with pytest.raises(KeyError):
        memory.update_feedback("missing", "BENIGN")


def test_append_and_duplicate_rejected():
    memory = AssociativeMemory().fit(sparse.csr_matrix([[1, 0]]), pd.DataFrame({"alert_id": ["a"]}))
    memory.add(sparse.csr_matrix([[0, 1]]), pd.DataFrame({"alert_id": ["b"]}))
    assert len(memory.metadata) == 2
    with pytest.raises(ValueError, match="unique"):
        memory.add(sparse.csr_matrix([[1, 1]]), pd.DataFrame({"alert_id": ["b"]}))
    assert len(memory.metadata) == 2


def test_leave_one_out_novelty_and_unknown_distance():
    x = sparse.csr_matrix([[1, 1, 0, 0], [1, 0, 1, 0]])
    detector = NearestNeighborNovelty().fit(x, ["a", "b"])
    assert detector.threshold_ == pytest.approx(2 / 3)
    result = detector.score(sparse.csr_matrix([[0, 0, 0, 1], [1, 1, 0, 0]]))
    assert result.is_novel.tolist() == [True, False]
    assert result.nearest_distance.tolist() == [1, 0]
    assert result.nearest_alert.iloc[1] == "a"


def test_training_duplicates_do_not_remove_all_neighbor_candidates():
    x = sparse.csr_matrix([[1, 0], [1, 0], [0, 1]])
    index = SparseIndex(x, "jaccard", batch_size=1)
    scores, positions = index.query(x, 10, exclude=np.arange(3))
    assert positions.shape == (3, 2)
    assert all(i not in row for i, row in enumerate(positions))
    assert scores[0, 0] == 1


def test_single_reference_cannot_calibrate():
    with pytest.raises(ValueError, match="two"):
        NearestNeighborNovelty().fit(sparse.csr_matrix([[1, 0]]))
