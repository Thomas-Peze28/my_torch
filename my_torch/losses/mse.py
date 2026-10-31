"""
Mean squared error loss
"""

import numpy as np

from my_torch.losses.loss import Loss


class MSELoss(Loss):
    """ Mean Squared Error (MSE) loss.
    
    Measure how far the predicted numbers are from the expected ones.

    The loss is the average of (prediction - target) squared, over every
    value of the batch. Squaring makes every gap positive, and punishes
    big gaps much more than small ones.
    """

    def __init__(self):
        self._last_gaps = None

    def forward(self, predictions, targets):
        """Measure the average squared gap.

        Args:
            predictions: Outputs of the network, shape (batch, outputs).
            targets: Expected outputs, same shape as predictions.

        Returns:
            The average squared gap, as a float.
        """
        self._last_gaps = predictions - targets
        return float(np.mean(self._last_gaps ** 2))

    def backward(self):
        """Return the error on each prediction.

        The derivative of gap squared is 2 * gap, divided by the number of
        values because the loss is an average.

        Returns:
            Array with the same shape as the predictions.
        """
        return 2.0 * self._last_gaps / self._last_gaps.size
