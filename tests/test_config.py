"""
Network configuration tests
"""

import json
from pathlib import Path

import numpy as np
import pytest

from my_torch import Linear, Network, ReLU, Softmax

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def small_description(**changes):
    description = {
        "layers": [
            {"type": "Linear", "input_size": 3, "output_size": 4},
            {"type": "ReLU"},
            {"type": "Linear", "input_size": 4, "output_size": 2},
            {"type": "Softmax"},
        ],
        "settings": {"learning_rate": 0.1, "epochs": 5},
    }
    description.update(changes)
    return description


def test_from_config_builds_the_described_layers():
    network = Network.from_config(small_description())
    assert [type(layer) for layer in network.layers] == [Linear, ReLU, Linear, Softmax]
    assert network.layers[0].weights.shape == (4, 3)
    assert network.layers[2].weights.shape == (2, 4)


def test_from_config_keeps_the_settings():
    assert Network.from_config(small_description()).settings == {"learning_rate": 0.1, "epochs": 5}


def test_from_config_settings_are_optional():
    description = small_description()
    del description["settings"]
    assert Network.from_config(description).settings == {}


def test_from_config_with_the_same_seed_gives_the_same_weights():
    first = Network.from_config(small_description(), seed=7)
    second = Network.from_config(small_description(), seed=7)
    for (first_value, _), (second_value, _) in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(first_value, second_value)


def test_from_config_with_another_seed_gives_other_weights():
    first = Network.from_config(small_description(), seed=7)
    second = Network.from_config(small_description(), seed=8)
    assert not np.array_equal(first.layers[0].weights, second.layers[0].weights)


def test_from_config_gives_each_linear_layer_its_own_weights():
    description = small_description(layers=[{"type": "Linear", "input_size": 3, "output_size": 3},
                                             {"type": "Linear", "input_size": 3, "output_size": 3}])
    network = Network.from_config(description, seed=0)
    assert not np.array_equal(network.layers[0].weights, network.layers[1].weights)


@pytest.mark.parametrize("description, message", [
    ([], "JSON object"),
    ({}, "layers"),
    ({"layers": []}, "layers"),
    (small_description(settings=[1, 2]), "settings"),
    (small_description(layers=[{"input_size": 3}]), "type"),
    (small_description(layers=[{"type": "Conv"}]), "Unknown layer type"),
    (small_description(layers=[{"type": "Linear", "size": 3}]), "Wrong settings"),
    (small_description(layers=[{"type": "Linear", "input_size": 3, "output_size": 0}]), "positive"),
    (small_description(layers=[{"type": "Linear", "input_size": 3, "output_size": 2.5}]), "positive"),
    (small_description(layers=[{"type": "Linear", "input_size": 3, "output_size": 4},
                               {"type": "ReLU"},
                               {"type": "Linear", "input_size": 8, "output_size": 2}]), "gets 8 inputs"),
])
def test_from_config_rejects_an_invalid_description(description, message):
    with pytest.raises(ValueError, match=message):
        Network.from_config(description)


@pytest.mark.parametrize("name", ["basic_network.conf", "chess_network.conf"])
def test_example_configurations_build_a_chess_network(name):
    description = json.loads((CONFIGS / name).read_text())
    network = Network.from_config(description, seed=0)
    probabilities = network.forward(np.zeros((2, 769)))
    assert probabilities.shape == (2, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    assert network.settings["loss"] == "CrossEntropyLoss"
