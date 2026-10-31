"""
Neural network
"""

import numpy as np

from my_torch.layers import Layer


class Network:
    """A chain of layers, called one after the other.

    Example:
        network = Network([Linear(769, 256), ReLU(), Linear(256, 5), Softmax()])
        
        probabilities = network.forward(boards)

    Args:
        layers: The layers of the network, from the first to the last.

    Attributes:
        layers: The list of layers.

    Raises:
        ValueError: If the list of layers is empty.
    """

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("A network needs at least one layer")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Pass the inputs through every layer, from the first to the last.

        Args:
            inputs: Array of shape (batch, inputs of the first layer).

        Returns:
            The outputs of the last layer.
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Send the error back through every layer, from the last to the first.

        Must be called after forward(), on the same batch. On the way, every
        layer with weights stores its gradients.

        Args:
            output_error: Error on the outputs of the last layer, usually
                given by a loss.

        Returns:
            The error on the inputs of the first layer.
        """
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """List the values every layer learns, each with its gradient.

        Returns:
            The (value, gradient) pairs of every layer, from the first
            layer to the last.
        """
        pairs = []
        for layer in self.layers:
            pairs.extend(layer.parameters())
        return pairs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Give the most likely answer for every example.

        Args:
            inputs: Array of shape (batch, inputs of the first layer).

        Returns:
            Array of shape (batch,): for each example, the index of the
            output with the biggest value.
        """
        return self.forward(inputs).argmax(axis=1)
