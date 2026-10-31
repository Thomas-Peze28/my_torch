"""
Network tests
"""

import numpy as np
import pytest

from my_torch.activations import ReLU, Softmax
from my_torch.layers import Linear
from my_torch.losses import CrossEntropyLoss
from my_torch.network import Network


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
