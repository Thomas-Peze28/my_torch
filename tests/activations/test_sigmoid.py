"""
Sigmoid activation tests
"""

import numpy as np

from my_torch.activations import Sigmoid


def test_sigmoid_known_values():
    outputs = Sigmoid().forward(np.array([[0.0, 2.0, -2.0]]))
    np.testing.assert_allclose(outputs, [[0.5, 0.880797, 0.119203]], atol=1e-6)


def test_sigmoid_keeps_the_shape():
    inputs = np.zeros((4, 3))
    assert Sigmoid().forward(inputs).shape == (4, 3)


def test_sigmoid_huge_inputs_do_not_overflow():
    with np.errstate(all="raise"):
        outputs = Sigmoid().forward(np.array([[-1e6, 1e6]]))
    np.testing.assert_allclose(outputs, [[0.0, 1.0]], atol=1e-12)


def test_sigmoid_backward_multiplies_the_error_by_the_hesitation():
    sigmoid = Sigmoid()
    sigmoid.forward(np.array([[0.0]]))           # sigmoid(0) = 0.5
    input_error = sigmoid.backward(np.array([[2.0]]))
    np.testing.assert_allclose(input_error, [[2.0 * 0.5 * 0.5]])   # 0.5
