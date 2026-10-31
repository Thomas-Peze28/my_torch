"""
SGD optimizer tests
"""

import numpy as np
import pytest

from my_torch.activations import ReLU, Sigmoid, Softmax
from my_torch.layers import Linear
from my_torch.losses import CrossEntropyLoss
from my_torch.network import Network
from my_torch.optimizers import SGD


def test_linear_parameters_are_its_own_arrays_with_their_gradients():
    linear = Linear(2, 3)
    (weights, weights_gradient), (bias, bias_gradient) = linear.parameters()
    assert weights is linear.weights
    assert weights_gradient is linear.weights_gradient
    assert bias is linear.bias
    assert bias_gradient is linear.bias_gradient


def test_activations_have_no_parameters():
    for activation in (Sigmoid(), ReLU(), Softmax()):
        assert activation.parameters() == []


def test_network_parameters_gather_every_layer():
    network = Network([Linear(3, 4), ReLU(), Linear(4, 2), Softmax()])
    assert len(network.parameters()) == 4


def test_sgd_step_moves_each_value_against_its_gradient():
    linear = Linear(2, 1)
    linear.weights = np.array([[1.0, 2.0]])
    linear.bias = np.array([0.5])
    linear.weights_gradient = np.array([[10.0, -20.0]])
    linear.bias_gradient = np.array([4.0])

    SGD(Network([linear]), learning_rate=0.1).step()

    np.testing.assert_allclose(linear.weights, [[0.0, 4.0]])
    np.testing.assert_allclose(linear.bias, [0.1])


def test_sgd_step_lowers_the_loss():
    network = Network([Linear(2, 4, seed=1), Sigmoid(), Linear(4, 2, seed=2), Softmax()])
    loss = CrossEntropyLoss()
    optimizer = SGD(network, learning_rate=0.1)
    inputs = np.array([[0.0, 1.0], [1.0, 0.0]])
    targets = np.array([[0.0, 1.0], [1.0, 0.0]])

    loss_before = loss.forward(network.forward(inputs), targets)
    network.backward(loss.backward())
    optimizer.step()
    loss_after = loss.forward(network.forward(inputs), targets)

    assert loss_after < loss_before


def test_sgd_learning_rate_must_be_positive():
    with pytest.raises(ValueError):
        SGD(Network([Linear(2, 1)]), learning_rate=0.0)
