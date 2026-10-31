"""
Loss base class
"""

from abc import ABC, abstractmethod


class Loss(ABC):
    """Base class of every loss: a measure of how wrong the network is.

    - forward(): compare the predictions with the expected answers and
      return one number, the loss. Lower is better, 0 is perfect.
    - backward(): return the error on each prediction, to start the
      backward pass of the network.
    """

    @abstractmethod
    def forward(self, predictions, targets):
        """Measure how wrong the predictions are.

        Args:
            predictions: Outputs of the network, shape (batch, outputs).
            targets: Expected outputs, same shape as predictions.

        Returns:
            The average loss over the batch, as a float.
        """

    @abstractmethod
    def backward(self):
        """Return the error on each prediction.

        Must be called after forward(), on the same batch.

        Returns:
            Array with the same shape as the predictions.
        """
