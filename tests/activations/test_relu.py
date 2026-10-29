"""
ReLU activation tests
"""

import numpy as np

from my_torch.activations import ReLU


def test_relu_keeps_positive_values_and_zeroes_negative_ones():
    outputs = ReLU().forward(np.array([[-2.0, 0.0, 3.0]]))
    np.testing.assert_array_equal(outputs, [[0.0, 0.0, 3.0]])


def test_relu_keeps_the_shape():
    inputs = np.zeros((4, 3))
    assert ReLU().forward(inputs).shape == (4, 3)


def test_relu_does_not_change_its_inputs():
    inputs = np.array([[-1.0, 2.0]])
    ReLU().forward(inputs)
    np.testing.assert_array_equal(inputs, [[-1.0, 2.0]])


def test_relu_backward_lets_the_error_through_only_where_the_input_was_positive():
    relu = ReLU()
    relu.forward(np.array([[-2.0, 0.0, 3.0]]))
    input_error = relu.backward(np.array([[5.0, 5.0, 5.0]]))
    np.testing.assert_array_equal(input_error, [[0.0, 0.0, 5.0]])
