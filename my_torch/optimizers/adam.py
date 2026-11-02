"""
Adam optimizer
"""

import numpy as np

from my_torch.network import Network
from my_torch.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """Adam: give every value its own step, from two memories.

    For each value of the network, Adam remembers:
    - the trend of its recent gradients (momentum), so back-and-forth
      gradients cancel out and only the real direction is kept;
    - the usual size of its gradients, so values with big gradients take
      calmer steps and values with small or rare gradients bigger ones.
    In the end every value moves by about learning_rate at each step.

    Args:
        network: The network whose values are changed.
        learning_rate: Size of each step. 0.001 is the usual value, much
            smaller than for SGD.
        beta1: How long the trend is remembered (0.9 is standard).
        beta2: How long the size of the gradients is remembered (0.999
            is standard).
        epsilon: Tiny number added so Adam never divides by 0.

    Raises:
        ValueError: If the learning rate or epsilon is not positive, or a
            beta is not between 0 (included) and 1 (excluded).
    """

    def __init__(self, network: Network, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        if learning_rate <= 0:
            raise ValueError("The learning rate must be positive")
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("beta1 and beta2 must be between 0 (included) and 1 (excluded)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.network = network
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = 0
        self.trends = [np.zeros_like(value) for value, _ in network.parameters()]
        self.sizes = [np.zeros_like(value) for value, _ in network.parameters()]

    def step(self) -> None:
        """Update the two memories, then move every value of the network once.

        At the start the memories are still 0, so they are too small: they
        are divided by (1 - beta ** steps) to correct them. This correction
        fades away after a few hundred steps.
        """
        self.steps += 1
        trend_fix = 1 - self.beta1 ** self.steps
        size_fix = 1 - self.beta2 ** self.steps
        for (value, gradient), trend, size in zip(self.network.parameters(), self.trends, self.sizes):
            trend *= self.beta1
            trend += (1 - self.beta1) * gradient
            size *= self.beta2
            size += (1 - self.beta2) * gradient ** 2
            value -= self.learning_rate * (trend / trend_fix) / (np.sqrt(size / size_fix) + self.epsilon)
