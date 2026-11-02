"""
Adam optimizer tests
"""

import numpy as np
import pytest

from my_torch import Adam, CrossEntropyLoss, Linear, Network, ReLU, Softmax


def network_with(weights, weights_gradient, bias=0.0, bias_gradient=0.0):
    """A one-neuron network whose values and gradients are chosen by the test."""
    linear = Linear(len(weights), 1)
    linear.weights[...] = [weights]
    linear.weights_gradient[...] = [weights_gradient]
    linear.bias[...] = bias
    linear.bias_gradient[...] = bias_gradient
    return Network([linear]), linear


def test_first_step_moves_each_value_by_the_learning_rate_against_its_gradient():
    network, linear = network_with([1.0, 2.0, 3.0], [5.0, -0.002, 300.0], bias_gradient=-1.0)
    Adam(network, learning_rate=0.1).step()
    np.testing.assert_allclose(linear.weights, [[0.9, 2.1, 2.9]], atol=1e-6)
    np.testing.assert_allclose(linear.bias, [0.1])


def test_big_and_small_gradients_give_steps_of_the_same_size():
    network, linear = network_with([0.0, 0.0], [1000.0, 0.001])
    optimizer = Adam(network, learning_rate=0.01)
    for _ in range(20):
        optimizer.step()
    big_move, small_move = -linear.weights[0]
    assert big_move == pytest.approx(small_move, rel=1e-3)
    assert big_move == pytest.approx(20 * 0.01, rel=1e-3)


def test_zero_gradient_does_not_move_the_value():
    network, linear = network_with([4.0], [0.0])
    optimizer = Adam(network)
    for _ in range(5):
        optimizer.step()
    np.testing.assert_array_equal(linear.weights, [[4.0]])


def test_back_and_forth_gradients_cancel_out():
    network, linear = network_with([0.0, 0.0], [0.0, 0.0])
    optimizer = Adam(network, learning_rate=0.01)
    for step in range(200):
        linear.weights_gradient[...] = [[1.0, 1.0 if step % 2 else -1.0]]
        optimizer.step()
    steady, zigzag = -linear.weights[0]
    assert steady == pytest.approx(2.0, rel=0.01)
    assert abs(zigzag) < steady / 20


def test_each_value_keeps_its_own_memories():
    network = Network([Linear(3, 4, seed=1), ReLU(), Linear(4, 2, seed=2)])
    optimizer = Adam(network)
    shapes = [value.shape for value, _ in network.parameters()]
    assert [trend.shape for trend in optimizer.trends] == shapes
    assert [size.shape for size in optimizer.sizes] == shapes
    assert optimizer.steps == 0


def test_step_changes_the_arrays_of_the_network_in_place():
    network, linear = network_with([1.0], [1.0])
    weights = linear.weights
    Adam(network).step()
    assert linear.weights is weights


def test_step_lowers_the_loss():
    network = Network([Linear(2, 4, seed=1, init="he"), ReLU(), Linear(4, 2, seed=2, init="he"), Softmax()])
    loss = CrossEntropyLoss()
    inputs = np.array([[0.0, 1.0], [1.0, 0.0]])
    targets = np.array([[0.0, 1.0], [1.0, 0.0]])
    loss_before = loss.forward(network.forward(inputs), targets)
    network.backward(loss.backward())
    Adam(network, learning_rate=0.01).step()
    assert loss.forward(network.forward(inputs), targets) < loss_before


@pytest.mark.parametrize("settings, message", [
    ({"learning_rate": 0}, "learning rate"),
    ({"learning_rate": -0.1}, "learning rate"),
    ({"beta1": 1.0}, "beta"),
    ({"beta2": -0.1}, "beta"),
    ({"epsilon": 0}, "epsilon"),
])
def test_invalid_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        Adam(Network([Linear(2, 1)]), **settings)
