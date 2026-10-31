"""
Network saving and loading tests
"""

import json

import numpy as np

from my_torch.activations import ReLU, Sigmoid, Softmax
from my_torch.layers import Linear
from my_torch.network import Network


def make_network(settings=None):
    return Network([Linear(3, 4, seed=1), ReLU(), Linear(4, 2, seed=2), Softmax()], settings)


def read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        architecture = json.loads(str(archive["architecture"]))
        arrays = {name: archive[name] for name in archive.files if name != "architecture"}
    return architecture, arrays


def test_activations_describe_themselves_with_their_type():
    assert ReLU().config() == {"type": "ReLU"}
    assert Sigmoid().config() == {"type": "Sigmoid"}
    assert Softmax().config() == {"type": "Softmax"}


def test_linear_describes_itself_with_its_sizes():
    assert Linear(769, 256).config() == {"type": "Linear", "input_size": 769, "output_size": 256}


def test_network_keeps_its_settings():
    assert make_network().settings == {}
    settings = {"learning_rate": 0.1, "epochs": 20}
    assert make_network(settings).settings == settings


def test_save_keeps_the_exact_file_name(tmp_path):
    path = tmp_path / "my_torch_network.nn"
    make_network().save(str(path))
    assert [file.name for file in tmp_path.iterdir()] == ["my_torch_network.nn"]


def test_save_writes_the_architecture_and_the_settings(tmp_path):
    path = tmp_path / "network.nn"
    make_network({"epochs": 20}).save(str(path))

    architecture, _ = read_archive(path)

    assert architecture == {
        "format": "my_torch",
        "version": 1,
        "layers": [
            {"type": "Linear", "input_size": 3, "output_size": 4},
            {"type": "ReLU"},
            {"type": "Linear", "input_size": 4, "output_size": 2},
            {"type": "Softmax"},
        ],
        "settings": {"epochs": 20},
    }


def test_save_writes_the_weights_of_every_linear_layer(tmp_path):
    path = tmp_path / "network.nn"
    network = make_network()
    network.save(str(path))

    _, arrays = read_archive(path)

    assert sorted(arrays) == ["layer_0_param_0", "layer_0_param_1",
                              "layer_2_param_0", "layer_2_param_1"]
    np.testing.assert_array_equal(arrays["layer_0_param_0"], network.layers[0].weights)
    np.testing.assert_array_equal(arrays["layer_0_param_1"], network.layers[0].bias)
    np.testing.assert_array_equal(arrays["layer_2_param_0"], network.layers[2].weights)
    np.testing.assert_array_equal(arrays["layer_2_param_1"], network.layers[2].bias)
