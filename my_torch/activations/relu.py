"""
ReLU activation
"""

import numpy as np

from my_torch.layers import Layer


class ReLU(Layer):
    """Keep positive numbers and turn negative numbers into 0.

    relu(x) = max(0, x). It is the usual activation of hidden layers:
    it is fast, and the error goes back unchanged through positive values.
    """

    def __init__(self) -> None:
        self._last_positive: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply ReLU to every value.

        Args:
            inputs: Array of any shape.

        Returns:
            Array of the same shape, with negative values replaced by 0.
        """
        self._last_positive = inputs > 0
        return np.where(self._last_positive, inputs, 0.0)

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Send the error back through ReLU.

        The error goes back unchanged where the input was positive, and is
        blocked (set to 0) where the input was negative or 0.

        Args:
            output_error: Error on the outputs, same shape as the outputs.

        Returns:
            Error on the inputs, same shape as the inputs.
        """
        return np.where(self._last_positive, output_error, 0.0)
