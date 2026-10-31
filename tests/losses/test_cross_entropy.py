"""
Cross-entropy loss tests
"""

import math

import numpy as np

from my_torch.activations import Softmax
from my_torch.losses import CrossEntropyLoss


def test_cross_entropy_is_zero_when_sure_of_the_right_answer():
    loss = CrossEntropyLoss().forward(np.array([[0.0, 1.0, 0.0]]),
                                      np.array([[0.0, 1.0, 0.0]]))
    assert loss == 0.0


def test_cross_entropy_known_value():
    loss = CrossEntropyLoss().forward(np.array([[0.7, 0.2, 0.1]]),
                                      np.array([[1.0, 0.0, 0.0]]))
    assert math.isclose(loss, -math.log(0.7))


def test_cross_entropy_is_the_average_of_the_batch():
    loss = CrossEntropyLoss().forward(np.array([[0.5, 0.5], [1.0, 0.0]]),
                                      np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert math.isclose(loss, (math.log(2) + 0.0) / 2)


def test_cross_entropy_zero_probability_on_the_right_answer_does_not_crash():
    with np.errstate(all="raise"):
        loss = CrossEntropyLoss().forward(np.array([[1.0, 0.0]]),
                                          np.array([[0.0, 1.0]]))
    assert math.isfinite(loss)
    assert loss > 20


def test_cross_entropy_backward_known_values():
    cross_entropy = CrossEntropyLoss()
    cross_entropy.forward(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(cross_entropy.backward(), [[-2.0, 0.0]])


def test_cross_entropy_after_softmax_gives_probabilities_minus_targets():
    scores = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    targets = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    softmax = Softmax()
    cross_entropy = CrossEntropyLoss()
    probabilities = softmax.forward(scores)
    cross_entropy.forward(probabilities, targets)
    score_error = softmax.backward(cross_entropy.backward())

    np.testing.assert_allclose(score_error, (probabilities - targets) / 2)
