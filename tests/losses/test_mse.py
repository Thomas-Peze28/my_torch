"""
Mean squared error loss tests
"""

import math

import numpy as np

from my_torch.losses import MSELoss


def test_mse_is_zero_when_every_prediction_is_right():
    loss = MSELoss().forward(np.array([[1.0, 0.0], [0.5, 2.0]]),
                             np.array([[1.0, 0.0], [0.5, 2.0]]))
    assert loss == 0.0


def test_mse_known_value():
    loss = MSELoss().forward(np.array([[1.0, 2.0], [3.0, 4.0]]),
                             np.array([[0.0, 2.0], [3.0, 1.0]]))
    assert math.isclose(loss, (1 + 0 + 0 + 9) / 4)


def test_mse_punishes_big_gaps_much_more():
    small = MSELoss().forward(np.array([[1.0]]), np.array([[0.0]]))
    big = MSELoss().forward(np.array([[3.0]]), np.array([[0.0]]))
    assert big == 9 * small


def test_mse_backward_known_values():
    mse = MSELoss()
    mse.forward(np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[0.0, 2.0], [3.0, 1.0]]))
    np.testing.assert_allclose(mse.backward(), [[0.5, 0.0],
                                                [0.0, 1.5]])
