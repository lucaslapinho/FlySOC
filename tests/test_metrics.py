import numpy as np
import pytest
from scipy import sparse

from flysoc.clustering import deduplicate
from flysoc.metrics import cluster_metrics, novelty_metrics, retrieval_metrics


def test_retrieval_denominators_and_unseen_exclusion():
    reference = np.array(["a", "a", "b", "c"])
    neighbors = np.array([[0, 2, 1], [0, 1, 2]])
    result = retrieval_metrics(neighbors, np.array(["a", "unseen"]), reference, [1, 3])
    assert result["eligible_queries"] == 1
    assert result["excluded_unseen_queries"] == 1
    assert result["by_k"]["3"]["precision_at_k"] == pytest.approx(2 / 3)
    assert result["by_k"]["3"]["recall_at_k"] == 1
    assert result["by_k"]["3"]["map_at_k"] == pytest.approx((1 + 2 / 3) / 2)


def test_pairwise_false_merge_definition():
    result = cluster_metrics(np.array([0, 0, 0, 1]), np.array(["a", "a", "b", "c"]))
    assert result["cluster_purity"] == .75
    assert result["false_merge_rate"] == pytest.approx(2 / 3)
    assert result["reduction_percentage"] == 50


def test_dedup_groups_identical_and_keeps_unrelated():
    x = sparse.csr_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    assert deduplicate(x).tolist() == [0, 0, 1]


def test_novelty_metrics_known_example_and_single_class():
    result = novelty_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .7, .9]), .75)
    assert result["tpr"] == .5 and result["fpr"] == .5
    assert result["auroc"] == .75
    assert novelty_metrics(np.array([0, 0]), np.array([.1, .2]), .5)["auroc"] is None
