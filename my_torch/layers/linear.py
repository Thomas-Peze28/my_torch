"""
Linear layer
"""

import numpy as np

from my_torch.layers.layer import Layer


class Linear(Layer):
    """A layer of neurons where every neuron is linked to every input.

    Each neuron computes a weighted total of the inputs plus its bias.
    The weights are stored in one array, with one row per neuron.

    Args:
        input_size: Number of inputs of each neuron.
        output_size: Number of neurons, so number of outputs.
        seed: Seed of the random starting weights, to get the same
            network every time. None gives different weights each time.

    Attributes:
        weights: Array of shape (output_size, input_size), one row per neuron.
        bias: Array of shape (output_size,), one bias per neuron.
        weights_gradient: How to change each weight to reduce the error,
            same shape as weights. Filled by backward().
        bias_gradient: How to change each bias to reduce the error, same
            shape as bias. Filled by backward().

    Raises:
        ValueError: If a size is not a positive whole number.
    """

    def __init__(self, input_size: int, output_size: int, seed: int | None = None) -> None:
        for size in (input_size, output_size):
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"Layer sizes must be positive whole numbers, not {size!r}")
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-1.0, 1.0, size=(output_size, input_size))
        self.bias = np.zeros(output_size)
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        self._last_inputs: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the total of every neuron, for every example.

        Args:
            inputs: Array of shape (batch, input_size), one row per example.

        Returns:
            Array of shape (batch, output_size): the total of each neuron
            for each example.
        """
        self._last_inputs = inputs
        return inputs @ self.weights.T + self.bias

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Find how to change the weights, and send the error back.

        Like in the playground, a weight is blamed by its neuron's error
        times the input it received. The weights are not changed here:
        an optimizer uses the gradients to do it.

        Args:
            output_error: Error on each neuron, shape (batch, output_size).

        Returns:
            Error on each input, shape (batch, input_size), for the
            previous layer.
        """
        self.weights_gradient = output_error.T @ self._last_inputs
        self.bias_gradient = output_error.sum(axis=0)
        return output_error @ self.weights

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """List the weights and the bias, each with its gradient.

        Returns:
            [(weights, weights_gradient), (bias, bias_gradient)].
        """
        return [(self.weights, self.weights_gradient), (self.bias, self.bias_gradient)]

    def config(self) -> dict:
        """Describe the layer, so that it can be saved and built again.

        Returns:
            {"type": "Linear", "input_size": ..., "output_size": ...}, read
            from the shape of the weights.
        """
        output_size, input_size = self.weights.shape
        return {"type": "Linear", "input_size": input_size, "output_size": output_size}
