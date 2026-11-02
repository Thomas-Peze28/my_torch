"""
Network saving and loading tests
"""

import json

import numpy as np
import pytest

from my_torch.activations import ReLU, Sigmoid, Softmax
from my_torch.layers import Linear
from my_torch.losses import CrossEntropyLoss
from my_torch.network import Network
from my_torch.optimizers import SGD


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


def test_linear_describes_itself_with_its_sizes_and_init():
    assert Linear(769, 256).config() == {"type": "Linear", "input_size": 769, "output_size": 256, "init": "uniform"}


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
            {"type": "Linear", "input_size": 3, "output_size": 4, "init": "uniform"},
            {"type": "ReLU"},
            {"type": "Linear", "input_size": 4, "output_size": 2, "init": "uniform"},
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


def saved_copy(network, tmp_path):
    path = tmp_path / "network.nn"
    network.save(str(path))
    return Network.load(str(path))


def write_archive(path, architecture, arrays=None):
    with open(path, "wb") as file:
        np.savez_compressed(file, architecture=np.array(json.dumps(architecture)), **(arrays or {}))


def valid_architecture():
    return {"format": "my_torch", "version": 1.0,
            "layers": [{"type": "Linear", "input_size": 2, "output_size": 3}, {"type": "Softmax"}],
            "settings": {}}


def test_load_gives_the_same_layers_weights_and_settings(tmp_path):
    network = make_network({"learning_rate": 0.1})
    loaded = saved_copy(network, tmp_path)

    assert [layer.config() for layer in loaded.layers] == [layer.config() for layer in network.layers]
    for (value, _), (loaded_value, _) in zip(network.parameters(), loaded.parameters()):
        np.testing.assert_array_equal(loaded_value, value)
    assert loaded.settings == {"learning_rate": 0.1}


def test_load_gives_the_same_predictions(tmp_path):
    network = make_network()
    loaded = saved_copy(network, tmp_path)
    inputs = np.random.default_rng(0).normal(size=(20, 3))
    np.testing.assert_array_equal(loaded.forward(inputs), network.forward(inputs))


def test_a_trained_xor_network_still_answers_right_after_loading(tmp_path):
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = np.eye(2)[[0, 1, 1, 0]]
    network = Network([Linear(2, 4, seed=0), Sigmoid(), Linear(4, 2, seed=1000), Softmax()])
    network.train(inputs, targets, CrossEntropyLoss(), SGD(network, 0.5), 2000, 4, seed=0)

    loaded = saved_copy(network, tmp_path)

    np.testing.assert_array_equal(loaded.predict(inputs), [0, 1, 1, 0])


def test_a_loaded_network_can_keep_training(tmp_path):
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = np.eye(2)[[0, 1, 1, 0]]
    loaded = saved_copy(Network([Linear(2, 4, seed=0), Sigmoid(), Linear(4, 2, seed=1000), Softmax()]), tmp_path)
    history = loaded.train(inputs, targets, CrossEntropyLoss(), SGD(loaded, 0.5), 300, 4, seed=0)
    assert history[-1] < history[0]


def test_load_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Network.load(str(tmp_path / "missing.nn"))


def test_load_a_text_file_raises_value_error(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(path))


def test_load_a_directory_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(tmp_path))


def test_load_an_archive_without_architecture_raises_value_error(tmp_path):
    path = tmp_path / "other.npz"
    with open(path, "wb") as file:
        np.savez(file, numbers=np.arange(3))
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(path))


@pytest.mark.parametrize("change", [
    {"format": "keras"},
    {"version": 2.0},
    {"layers": [{"type": "Conv", "size": 3}]},
    {"layers": [{"type": "Linear", "size": 3}]},
])
def test_load_an_invalid_architecture_raises_value_error(tmp_path, change):
    path = tmp_path / "network.nn"
    write_archive(path, valid_architecture() | change,
                  {"layer_0_param_0": np.zeros((3, 2)), "layer_0_param_1": np.zeros(3)})
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(path))


def test_load_missing_weights_raises_value_error(tmp_path):
    path = tmp_path / "network.nn"
    write_archive(path, valid_architecture(), {"layer_0_param_0": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(path))


def test_load_weights_of_the_wrong_shape_raises_value_error(tmp_path):
    path = tmp_path / "network.nn"
    write_archive(path, valid_architecture(),
                  {"layer_0_param_0": np.zeros((5, 5)), "layer_0_param_1": np.zeros(3)})
    with pytest.raises(ValueError, match="Wrong shape"):
        Network.load(str(path))


def test_load_an_empty_file_raises_value_error(tmp_path):
    path = tmp_path / "empty.nn"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(path))


def test_load_a_cut_file_raises_value_error(tmp_path):
    full = tmp_path / "full.nn"
    make_network().save(str(full))
    cut = tmp_path / "cut.nn"
    cut.write_bytes(full.read_bytes()[:500])
    with pytest.raises(ValueError, match="not a my_torch network file"):
        Network.load(str(cut))
