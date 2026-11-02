"""
Starting weights tests
"""

import numpy as np
import pytest

from my_torch import Linear, Network
from my_torch.layers.initializers import INITIALIZERS


@pytest.mark.parametrize("init, spread", [
    ("uniform", np.sqrt(1 / 3)),
    ("lecun", np.sqrt(1 / 400)),
    ("xavier", np.sqrt(2 / (400 + 300))),
    ("he", np.sqrt(2 / 400)),
])
def test_each_init_gives_weights_of_the_expected_size(init, spread):
    weights = Linear(400, 300, seed=0, init=init).weights
    assert weights.shape == (300, 400)
    assert abs(weights.mean()) < spread / 10
    assert weights.std() == pytest.approx(spread, rel=0.02)


def test_bigger_layers_get_smaller_weights():
    small = Linear(10, 50, seed=0, init="he").weights.std()
    big = Linear(1000, 50, seed=0, init="he").weights.std()
    assert big < small / 5


def test_xavier_stays_between_its_limits():
    limit = np.sqrt(6 / (30 + 20))
    weights = Linear(30, 20, seed=0, init="xavier").weights
    assert weights.min() >= -limit and weights.max() <= limit


@pytest.mark.parametrize("init", list(INITIALIZERS))
def test_the_same_seed_gives_the_same_weights(init):
    np.testing.assert_array_equal(Linear(5, 4, seed=7, init=init).weights,
                                  Linear(5, 4, seed=7, init=init).weights)


def test_bias_starts_at_zero_whatever_the_init():
    for init in INITIALIZERS:
        np.testing.assert_array_equal(Linear(5, 4, seed=1, init=init).bias, np.zeros(4))


def test_uniform_is_the_default_and_keeps_the_old_weights():
    rng = np.random.default_rng(3)
    np.testing.assert_array_equal(Linear(6, 2, seed=3).weights, rng.uniform(-1.0, 1.0, size=(2, 6)))
    assert Linear(6, 2).init == "uniform"


def test_unknown_init_is_refused():
    with pytest.raises(ValueError, match='Unknown init "zeros"'):
        Linear(3, 2, init="zeros")


def test_config_and_save_keep_the_init(tmp_path):
    network = Network([Linear(4, 3, seed=1, init="he"), Linear(3, 2, seed=2, init="xavier")])
    assert network.layers[0].config()["init"] == "he"
    network.save(str(tmp_path / "network.nn"))
    loaded = Network.load(str(tmp_path / "network.nn"))
    assert [layer.init for layer in loaded.layers] == ["he", "xavier"]


def test_from_config_chooses_the_init_of_each_layer():
    network = Network.from_config({"layers": [
        {"type": "Linear", "input_size": 4, "output_size": 3, "init": "he"},
        {"type": "ReLU"},
        {"type": "Linear", "input_size": 3, "output_size": 2},
    ]}, seed=0)
    assert [network.layers[0].init, network.layers[2].init] == ["he", "uniform"]


def test_from_config_refuses_an_unknown_init():
    with pytest.raises(ValueError, match="Unknown init"):
        Network.from_config({"layers": [{"type": "Linear", "input_size": 4, "output_size": 3, "init": "big"}]})
