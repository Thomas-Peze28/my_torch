"""
Layer base class
"""

from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class of every layer of a network.

    A network is a chain of layers. Each layer must be able to:
    - forward(): turn its inputs into outputs and pass them to the next layer;
    - backward(): take the error of its outputs and pass the error of its
      inputs back to the previous layer.
    """

    @abstractmethod
    def forward(self, inputs):
        """Compute the outputs of the layer.

        Args:
            inputs: Array of shape (batch, features).

        Returns:
            The outputs of the layer.
        """

    @abstractmethod
    def backward(self, output_error):
        """Send the error back through the layer.

        Must be called after forward(), on the same batch.

        Args:
            output_error: Error on the outputs of this layer, same shape as
                the outputs.

        Returns:
            Error on the inputs of this layer, same shape as the inputs.
        """
