"""
Cross-entropy loss
"""

import numpy as np

from my_torch.losses.loss import Loss

_SMALLEST_PROBABILITY = 1e-12


class CrossEntropyLoss(Loss):
    """Measure how wrong a choice between several answers is.

    The loss of one example is -log(probability given to the right answer):
    0 when the network is sure of the right answer, and bigger and bigger
    when it gives the right answer a small probability.

    The predictions must be probabilities, so the network must end with a
    Softmax layer. Unlike PyTorch, the softmax is not inside this loss.
    The targets are "one-hot": 1 for the right answer, 0 for the others.
    """

    def __init__(self) -> None:
        self._last_probabilities: np.ndarray | None = None
        self._last_targets: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Measure how wrong the probabilities are.

        Args:
            predictions: Probabilities, shape (batch, answers), each row
                adding up to 1.
            targets: One-hot answers, same shape as predictions.

        Returns:
            The average loss over the batch, as a float.
        """
        probabilities = np.clip(predictions, _SMALLEST_PROBABILITY, 1.0)
        self._last_probabilities = probabilities
        self._last_targets = targets
        losses = -np.sum(targets * np.log(probabilities), axis=1)
        return float(losses.mean())

    def backward(self) -> np.ndarray:
        """Return the error on each probability.

        The error is -target / probability, divided by the batch size
        because the loss is an average. Followed by Softmax.backward(),
        it becomes simply (probabilities - targets) / batch size.

        Returns:
            Array of shape (batch, answers).
        """
        batch_size = self._last_targets.shape[0]
        return -self._last_targets / self._last_probabilities / batch_size
