import numpy as np
import pytest

from flysoc.plotting import create_plots


@pytest.mark.parametrize("unseen", [None, np.array([False, False]), np.array([True, True])])
def test_unannotated_or_single_population_exports_do_not_fabricate_auc(tmp_path, unseen):
    metrics = {"retrieval": {}, "classification": {}, "novelty": {},
               "deduplication": {"fly": {"reduction_percentage": 0, "threshold": .9}},
               "performance": {"retrieval": {"fly": {"individual_queries": {"median_ms": 1}}}}}
    create_plots(tmp_path, metrics, {"fly": np.array([.2, .3])}, {"fly": np.array([.8, .7])}, unseen)
    assert (tmp_path / "alert_reduction.png").is_file()
    assert not (tmp_path / "novelty_comparison.png").exists()
