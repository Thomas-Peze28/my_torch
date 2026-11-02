"""
Network tests
"""

import numpy as np
import pytest

from my_torch.activations import ReLU, Sigmoid, Softmax
from my_torch.layers import Linear
from my_torch.losses import CrossEntropyLoss
from my_torch.network import Network
from my_torch.optimizers import SGD


def make_layers():
    return [Linear(3, 4, seed=1), ReLU(), Linear(4, 2, seed=2), Softmax()]


def test_network_forward_calls_every_layer_in_order():
    inputs = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])

    by_hand = inputs
    for layer in make_layers():
        by_hand = layer.forward(by_hand)

    network = Network(make_layers())
    np.testing.assert_allclose(network.forward(inputs), by_hand)


def test_network_ending_with_softmax_gives_rows_adding_up_to_1():
    network = Network(make_layers())
    probabilities = network.forward(np.ones((5, 3)))
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(5))


def test_network_needs_at_least_one_layer():
    with pytest.raises(ValueError):
        Network([])


def test_network_backward_fills_the_gradients_of_every_linear_layer():
    network = Network(make_layers())
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    loss = CrossEntropyLoss()

    probabilities = network.forward(np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]]))
    loss.forward(probabilities, targets)
    network.backward(loss.backward())

    for layer in (network.layers[0], network.layers[2]):
        assert np.any(layer.weights_gradient != 0)
        assert np.any(layer.bias_gradient != 0)


def test_network_backward_calls_every_layer_in_reverse_order():
    inputs = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    output_error = np.array([[0.3, -0.3], [-0.1, 0.1]])

    layers = make_layers()
    outputs = inputs
    for layer in layers:
        outputs = layer.forward(outputs)
    by_hand = output_error
    for layer in reversed(layers):
        by_hand = layer.backward(by_hand)

    network = Network(make_layers())
    network.forward(inputs)
    np.testing.assert_allclose(network.backward(output_error), by_hand)


def make_known_network():
    linear = Linear(2, 3)
    linear.weights = np.array([[1.0, 0.0],
                               [0.0, 1.0],
                               [-1.0, -1.0]])
    linear.bias = np.zeros(3)
    return Network([linear, Softmax()])


def test_network_predict_gives_the_index_of_the_biggest_output():
    network = make_known_network()
    inputs = np.array([[5.0, 1.0],
                       [1.0, 5.0],
                       [-3.0, -3.0]])
    np.testing.assert_array_equal(network.predict(inputs), [0, 1, 2])


def test_network_predict_gives_one_answer_per_example():
    network = Network(make_layers())
    assert network.predict(np.ones((7, 3))).shape == (7,)


def test_network_predict_matches_argmax_of_forward():
    network = Network(make_layers())
    inputs = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0], [2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(network.predict(inputs),
                                  network.forward(inputs).argmax(axis=1))


def test_network_predict_is_the_same_with_or_without_softmax():
    inputs = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0], [2.0, 2.0, 2.0]])
    with_softmax = Network(make_layers())
    without_softmax = Network(make_layers()[:-1])
    np.testing.assert_array_equal(with_softmax.predict(inputs),
                                  without_softmax.predict(inputs))


XOR_INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_TARGETS = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])


def train_xor(epochs, batch_size=2, seed=0):
    network = Network([Linear(2, 4, seed=1), Sigmoid(), Linear(4, 2, seed=2), Softmax()])
    return network.train(XOR_INPUTS, XOR_TARGETS, CrossEntropyLoss(),
                         SGD(network, learning_rate=0.5), epochs, batch_size, seed)["loss"]


def test_network_train_returns_one_loss_per_epoch():
    assert len(train_xor(epochs=7)) == 7


def test_network_train_lowers_the_loss():
    history = train_xor(epochs=300)
    assert history[-1] < history[0]


def test_network_train_with_the_same_seed_gives_the_same_history():
    assert train_xor(epochs=50, seed=3) == train_xor(epochs=50, seed=3)


def test_network_train_accepts_a_batch_bigger_than_the_examples():
    assert len(train_xor(epochs=5, batch_size=100)) == 5


@pytest.mark.parametrize("epochs, batch_size", [(0, 2), (5, 0), (-1, 2)])
def test_network_train_rejects_non_positive_settings(epochs, batch_size):
    with pytest.raises(ValueError):
        train_xor(epochs, batch_size)


def test_network_train_rejects_inputs_and_targets_of_different_sizes():
    network = Network([Linear(2, 2), Softmax()])
    with pytest.raises(ValueError):
        network.train(XOR_INPUTS, XOR_TARGETS[:3], CrossEntropyLoss(),
                      SGD(network, learning_rate=0.5), epochs=1, batch_size=2)
