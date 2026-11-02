"""
XOR learning tests

XOR answers 1 when its two inputs are different. A single layer cannot
learn it: these tests check that the whole library, put together, can.
"""

import math

import numpy as np
import pytest

from my_torch.activations import ReLU, Sigmoid, Softmax
from my_torch.layers import Linear
from my_torch.losses import CrossEntropyLoss, MSELoss
from my_torch.network import Network
from my_torch.optimizers import SGD, Adam

INPUTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_ANSWERS = np.array([0, 1, 1, 0])
AND_ANSWERS = np.array([0, 0, 0, 1])
OR_ANSWERS = np.array([0, 1, 1, 1])


def one_hot(answers):
    """Turn answer indexes into one-hot rows: 1 -> [0, 1]."""
    return np.eye(2)[answers]


def sigmoid_network(seed, hidden=4, last=Softmax):
    return Network([Linear(2, hidden, seed=seed), Sigmoid(),
                    Linear(hidden, 2, seed=seed + 1000), last()])


def relu_network(seed):
    return Network([Linear(2, 8, seed=seed), ReLU(),
                    Linear(8, 2, seed=seed + 1000), Softmax()])


SETUPS = {
    "sigmoid + cross-entropy": (sigmoid_network, CrossEntropyLoss, 0.5, 2000, 0.05),
    "relu + cross-entropy": (relu_network, CrossEntropyLoss, 0.1, 2000, 0.05),
    "sigmoid output + mse": (lambda seed: sigmoid_network(seed, last=Sigmoid), MSELoss, 2.0, 3000, 0.01),
    "softmax output + mse": (sigmoid_network, MSELoss, 2.0, 3000, 0.01),
}


def train(network, loss, learning_rate, epochs, batch_size=4, seed=0, answers=XOR_ANSWERS):
    return network.train(INPUTS, one_hot(answers), loss, SGD(network, learning_rate),
                         epochs, batch_size, seed)["loss"]


def right_answer_probabilities(network, answers=XOR_ANSWERS):
    return network.forward(INPUTS)[np.arange(len(answers)), answers]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("setup", SETUPS)
def test_xor_is_learned(setup, batch_size):
    make_network, make_loss, learning_rate, epochs, max_final_loss = SETUPS[setup]
    network = make_network(0)

    history = train(network, make_loss(), learning_rate, epochs, batch_size)

    np.testing.assert_array_equal(network.predict(INPUTS), XOR_ANSWERS)
    assert right_answer_probabilities(network).min() > 0.9
    assert history[-1] < max_final_loss
    assert history[-1] < history[0] / 10


@pytest.mark.parametrize("setup", ["sigmoid + cross-entropy", "relu + cross-entropy"])
def test_xor_is_learned_from_many_starting_points(setup):
    make_network, make_loss, learning_rate, epochs, _ = SETUPS[setup]
    for seed in range(10):
        network = make_network(seed)
        train(network, make_loss(), learning_rate, epochs, seed=seed)
        np.testing.assert_array_equal(network.predict(INPUTS), XOR_ANSWERS, err_msg=f"seed {seed}")


def test_xor_with_two_hidden_neurons_can_get_stuck():
    lucky = sigmoid_network(0, hidden=2)
    train(lucky, CrossEntropyLoss(), 0.5, 3000)
    np.testing.assert_array_equal(lucky.predict(INPUTS), XOR_ANSWERS)

    stuck = sigmoid_network(7, hidden=2)
    history = train(stuck, CrossEntropyLoss(), 0.5, 3000, seed=7)
    assert not np.array_equal(stuck.predict(INPUTS), XOR_ANSWERS)
    assert history[-1] > 0.3
    probabilities = stuck.forward(INPUTS)
    hesitant_rows = np.isclose(probabilities[:, 0], 0.5, atol=0.05)
    assert hesitant_rows.sum() == 2


def test_xor_cannot_be_learned_without_a_hidden_layer():
    network = Network([Linear(2, 2, seed=0), Softmax()])
    history = train(network, CrossEntropyLoss(), 0.5, 2000)
    right_answers = (network.predict(INPUTS) == XOR_ANSWERS).sum()
    assert right_answers <= 3
    assert math.isclose(history[-1], math.log(2), abs_tol=0.01)


@pytest.mark.parametrize("answers", [AND_ANSWERS, OR_ANSWERS], ids=["and", "or"])
def test_and_and_or_are_learned_without_a_hidden_layer(answers):
    network = Network([Linear(2, 2, seed=0), Softmax()])
    train(network, CrossEntropyLoss(), 0.5, 2000, answers=answers)
    np.testing.assert_array_equal(network.predict(INPUTS), answers)


def test_xor_training_is_reproducible():
    first, second = sigmoid_network(3), sigmoid_network(3)
    first_history = train(first, CrossEntropyLoss(), 0.5, 200, batch_size=2, seed=3)
    second_history = train(second, CrossEntropyLoss(), 0.5, 200, batch_size=2, seed=3)
    assert first_history == second_history
    for (first_value, _), (second_value, _) in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(first_value, second_value)


def test_xor_probabilities_are_valid():
    network = sigmoid_network(0)
    train(network, CrossEntropyLoss(), 0.5, 2000)
    probabilities = network.forward(INPUTS)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))


def test_xor_answers_do_not_depend_on_the_order_of_the_examples():
    network = sigmoid_network(0)
    train(network, CrossEntropyLoss(), 0.5, 2000)
    order = [3, 1, 0, 2]
    np.testing.assert_array_equal(network.predict(INPUTS[order]), XOR_ANSWERS[order])


def test_xor_answers_are_the_same_one_example_at_a_time():
    network = sigmoid_network(0)
    train(network, CrossEntropyLoss(), 0.5, 2000)
    one_by_one = [network.predict(INPUTS[i:i + 1])[0] for i in range(4)]
    np.testing.assert_array_equal(one_by_one, XOR_ANSWERS)


def test_xor_stays_learned_when_training_continues():
    network = sigmoid_network(0)
    loss = CrossEntropyLoss()
    first_history = train(network, loss, 0.5, 2000)
    second_history = train(network, loss, 0.5, 2000)
    np.testing.assert_array_equal(network.predict(INPUTS), XOR_ANSWERS)
    assert second_history[-1] <= first_history[-1]


@pytest.mark.parametrize("seed", range(5))
def test_xor_is_learned_with_adam(seed):
    network = relu_network(seed)
    history = network.train(INPUTS, one_hot(XOR_ANSWERS), CrossEntropyLoss(), Adam(network, learning_rate=0.01),
                            epochs=1000, batch_size=4, seed=seed)["loss"]
    assert history[-1] < 0.05
    assert (right_answer_probabilities(network) > 0.9).all()
