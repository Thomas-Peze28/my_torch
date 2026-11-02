"""
Starting weights of a Linear layer

Each function draws the weights of a layer at random, with a size chosen
from its number of inputs (and outputs), so the totals of the neurons are
neither too big nor too small when the training starts.
"""

from collections.abc import Callable

import numpy as np

Initializer = Callable[[np.random.Generator, int, int], np.ndarray]
"""A function that draws starting weights: (rng, input_size, output_size) -> weights."""


def uniform(rng: np.random.Generator, input_size: int, output_size: int) -> np.ndarray:
    """Weights between -1 and 1, whatever the size of the layer.

    Fine for very small layers, too big for large ones.
    """
    return rng.uniform(-1.0, 1.0, size=(output_size, input_size))


def lecun(rng: np.random.Generator, input_size: int, output_size: int) -> np.ndarray:
    """LeCun: normal weights with a spread of sqrt(1 / inputs).

    The more inputs a neuron adds up, the smaller its weights.
    """
    return rng.normal(0.0, np.sqrt(1.0 / input_size), size=(output_size, input_size))


def xavier(rng: np.random.Generator, input_size: int, output_size: int) -> np.ndarray:
    """Xavier (Glorot): uniform weights between -limit and limit.

    The limit is sqrt(6 / (inputs + outputs)), so the values going forward
    and the errors going back both keep the same size. Made for Sigmoid.
    """
    limit = np.sqrt(6.0 / (input_size + output_size))
    return rng.uniform(-limit, limit, size=(output_size, input_size))


def he(rng: np.random.Generator, input_size: int, output_size: int) -> np.ndarray:
    """He: normal weights with a spread of sqrt(2 / inputs).

    Like LeCun, twice as big, because ReLU turns half of the values into 0.
    Made for ReLU.
    """
    return rng.normal(0.0, np.sqrt(2.0 / input_size), size=(output_size, input_size))


INITIALIZERS: dict[str, Initializer] = {
    "uniform": uniform,
    "lecun": lecun,
    "xavier": xavier,
    "he": he,
}
