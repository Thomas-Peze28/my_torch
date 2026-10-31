"""
Stochastic gradient descent optimizer
"""

from my_torch.network import Network
from my_torch.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent (SGD): move every value against its gradient.

    Args:
        network: The network whose values are changed.
        learning_rate: Size of each correction. Too small learns slowly,
            too big jumps over the right values.

    Raises:
        ValueError: If the learning rate is not positive.
    """

    def __init__(self, network: Network, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("The learning rate must be positive")
        self.network = network
        self.learning_rate = learning_rate

    def step(self) -> None:
        """Move every value of the network once against its gradient."""
        for value, gradient in self.network.parameters():
            value -= self.learning_rate * gradient
