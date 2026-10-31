"""
Sigmoid activation
"""

import numpy as np

from my_torch.layers import Layer

_EXP_LIMIT = 500.0


class Sigmoid(Layer):
    """Squash every number between 0 and 1.

    sigmoid(x) = 1 / (1 + exp(-x)). Very negative inputs give almost 0,
    very positive inputs give almost 1, and 0 gives 0.5.
    """

    def __init__(self) -> None:
        self._last_outputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply the sigmoid to every value.

        Args:
            inputs: Array of any shape.

        Returns:
            Array of the same shape, with values between 0 and 1.
        """
        clipped = np.clip(inputs, -_EXP_LIMIT, _EXP_LIMIT)
        self._last_outputs = 1.0 / (1.0 + np.exp(-clipped))
        return self._last_outputs

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Send the error back through the sigmoid.

        The error is multiplied by how much the sigmoid "hesitates":
        outputs * (1 - outputs), which is its derivative.

        Args:
            output_error: Error on the outputs, same shape as the outputs.

        Returns:
            Error on the inputs, same shape as the inputs.
        """
        return output_error * self._last_outputs * (1.0 - self._last_outputs)
