"""
Linear layer tests
"""

import numpy as np

from my_torch.layers import Linear


def test_linear_has_one_row_of_weights_per_neuron():
    linear = Linear(2, 3)
    assert linear.weights.shape == (3, 2)
    assert linear.bias.shape == (3,)


def test_linear_output_shape_is_batch_by_neurons():
    linear = Linear(2, 3)
    outputs = linear.forward(np.zeros((4, 2)))
    assert outputs.shape == (4, 3)


def test_linear_computes_the_total_of_each_neuron():
    linear = Linear(2, 3)
    linear.weights = np.array([[1.0, 2.0],
                              [0.5, -1.0],
                              [0.0, 3.0]])
    linear.bias = np.array([0.0, 10.0, -1.0])

    outputs = linear.forward(np.array([[1.0, 1.0],
                                      [2.0, 0.0]]))

    np.testing.assert_allclose(outputs, [[3.0, 9.5, 2.0],
                                         [2.0, 11.0, -1.0]])


def test_linear_same_seed_gives_same_weights():
    first = Linear(5, 4, seed=42)
    second = Linear(5, 4, seed=42)
    np.testing.assert_array_equal(first.weights, second.weights)
