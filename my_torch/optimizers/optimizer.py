"""
Optimizer base class
"""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class of every optimizer: the tool that turns the knobs.

    After network.backward(), every value the network learns has a
    gradient. step() uses these gradients to change the values so that
    the error gets smaller.
    """

    @abstractmethod
    def step(self) -> None:
        """Change every value of the network once, using its gradient.

        Must be called after network.backward().
        """
