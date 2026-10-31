"""
Softmax activation tests
"""

import numpy as np

from my_torch.activations import Softmax


def test_softmax_known_values():
    outputs = Softmax().forward(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(outputs, [[0.090031, 0.244728, 0.665241]], atol=1e-6)


def test_softmax_rows_add_up_to_1():
    inputs = np.array([[1.0, -2.0, 0.5, 4.0], [0.0, 0.0, 0.0, 0.0]])
    outputs = Softmax().forward(inputs)
    np.testing.assert_allclose(outputs.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(outputs[1], [0.25, 0.25, 0.25, 0.25])


def test_softmax_huge_inputs_do_not_overflow():
    with np.errstate(over="raise", invalid="raise"):
        outputs = Softmax().forward(np.array([[1000.0, 1000.0], [-1000.0, 0.0]]))
    np.testing.assert_allclose(outputs, [[0.5, 0.5], [0.0, 1.0]], atol=1e-12)


def test_softmax_backward_known_values():
    softmax = Softmax()
    softmax.forward(np.array([[0.0, 0.0]]))
    input_error = softmax.backward(np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(input_error, [[0.25, -0.25]])


def test_softmax_backward_ignores_an_error_shared_by_every_answer():
    softmax = Softmax()
    softmax.forward(np.array([[1.0, 2.0, 3.0]]))
    input_error = softmax.backward(np.array([[7.0, 7.0, 7.0]]))
    np.testing.assert_allclose(input_error, [[0.0, 0.0, 0.0]], atol=1e-12)
