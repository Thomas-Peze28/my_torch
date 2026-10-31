"""
Softmax activation
"""

import numpy as np

from my_torch.layers import Layer


class Softmax(Layer):
    """Turn each row of numbers into probabilities that add up to 1.

    softmax(x)_i = exp(x_i) / sum(exp(x_j)). The biggest number gets the
    biggest probability. It is used on the last layer, to choose between
    several answers.
    """

    def __init__(self) -> None:
        self._last_probabilities: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply the softmax to every row (one row = one example).

        The biggest score of each row is subtracted first: the result is
        the same, and np.exp() never overflows.

        Args:
            inputs: Array of shape (batch, answers).

        Returns:
            Array of the same shape, where each row is positive and adds up to 1.
        """
        row_max = inputs.max(axis=1, keepdims=True)
        exps = np.exp(inputs - row_max)
        row_total = exps.sum(axis=1, keepdims=True)
        self._last_probabilities = exps / row_total
        return self._last_probabilities

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Send the error back through the softmax.

        The probabilities of a row always add up to 1: raising one lowers
        the others. So each input gets its own error minus the average
        error of its row, scaled by its probability.

        Args:
            output_error: Error on the outputs, shape (batch, answers).

        Returns:
            Error on the inputs, shape (batch, answers).
        """
        probabilities = self._last_probabilities
        average_error = (output_error * probabilities).sum(axis=1, keepdims=True)
        return probabilities * (output_error - average_error)
