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
    """

    def __init__(self, input_size, output_size, seed=None):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-1.0, 1.0, size=(output_size, input_size))
        self.bias = np.zeros(output_size)

    def forward(self, inputs):
        """Compute the total of every neuron, for every example.

        Args:
            inputs: Array of shape (batch, input_size), one row per example.

        Returns:
            Array of shape (batch, output_size): the total of each neuron
            for each example.
        """
        return inputs @ self.weights.T + self.bias

    def backward(self, output_error):
        """Send the error back through the layer."""
        raise NotImplementedError
