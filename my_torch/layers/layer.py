"""
Layer base class
"""

from abc import ABC, abstractmethod

import numpy as np

Parameter = tuple[np.ndarray, np.ndarray]
"""A value the network learns and its gradient: (value, gradient), same shape."""


class Layer(ABC):
    """Base class of every layer of a network.

    A network is a chain of layers. Each layer must be able to:
    - forward(): turn its inputs into outputs and pass them to the next layer;
    - backward(): take the error of its outputs and pass the error of its
      inputs back to the previous layer.
    """

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the outputs of the layer.

        Args:
            inputs: Array of shape (batch, features).

        Returns:
            The outputs of the layer.
        """

    @abstractmethod
    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Send the error back through the layer.

        Must be called after forward(), on the same batch.

        Args:
            output_error: Error on the outputs of this layer, same shape as
                the outputs.

        Returns:
            Error on the inputs of this layer, same shape as the inputs.
        """

    def parameters(self) -> list[Parameter]:
        """List the values the layer learns, each with its gradient.

        Layers without weights, like activations, have nothing to learn:
        they keep this default and return an empty list.

        Returns:
            A list of (value, gradient) pairs, where both arrays have the
            same shape.
        """
        return []

    def config(self) -> dict:
        """Describe the layer, so that it can be saved and built again.

        Layers without settings, like activations, keep this default and
        only give the name of their class.

        Returns:
            A dictionary with at least the "type" key, e.g. {"type": "ReLU"}.
        """
        return {"type": type(self).__name__}
