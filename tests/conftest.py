import pytest

from flysoc.config import load_config
from flysoc.synthetic import generate_alerts


@pytest.fixture
def config():
    cfg = load_config()
    cfg["features"].update(input_features=128, categorical_hash_dim=64, text_max_features=56)
    cfg["flyhash"].update(kc_dim=256, fan_in=8, top_k=16, batch_size=8)
    cfg["classification"]["forest_trees"] = 5
    return cfg


@pytest.fixture
def alerts(config):
    return generate_alerts(config, 300)
