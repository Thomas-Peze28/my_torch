"""
Neural network
"""

import json
from typing import TYPE_CHECKING

import numpy as np

from my_torch.layers import Layer
from my_torch.losses import Loss

if TYPE_CHECKING:
    from my_torch.optimizers import Optimizer

FILE_FORMAT = "my_torch"
FILE_VERSION = 1.0


class Network:
    """A chain of layers, called one after the other.

    Example:
        network = Network([Linear(769, 256), ReLU(), Linear(256, 5), Softmax()])
        
        probabilities = network.forward(boards)

    Args:
        layers: The layers of the network, from the first to the last.
        settings: Free settings saved with the network, e.g. how to train
            it. The library only stores them; they must be JSON-friendly
            (numbers, strings, lists, dictionaries).

    Attributes:
        layers: The list of layers.
        settings: The settings given at creation, or an empty dictionary.

    Raises:
        ValueError: If the list of layers is empty.
    """

    def __init__(self, layers: list[Layer], settings: dict | None = None) -> None:
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("A network needs at least one layer")
        self.settings = dict(settings) if settings else {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Pass the inputs through every layer, from the first to the last.

        Args:
            inputs: Array of shape (batch, inputs of the first layer).

        Returns:
            The outputs of the last layer.
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, output_error: np.ndarray) -> np.ndarray:
        """Send the error back through every layer, from the last to the first.

        Must be called after forward(), on the same batch. On the way, every
        layer with weights stores its gradients.

        Args:
            output_error: Error on the outputs of the last layer, usually
                given by a loss.

        Returns:
            The error on the inputs of the first layer.
        """
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """List the values every layer learns, each with its gradient.

        Returns:
            The (value, gradient) pairs of every layer, from the first
            layer to the last.
        """
        pairs = []
        for layer in self.layers:
            pairs.extend(layer.parameters())
        return pairs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Give the most likely answer for every example.

        Args:
            inputs: Array of shape (batch, inputs of the first layer).

        Returns:
            Array of shape (batch,): for each example, the index of the
            output with the biggest value.
        """
        return self.forward(inputs).argmax(axis=1)

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        loss: Loss,
        optimizer: Optimizer,
        epochs: int,
        batch_size: int,
        seed: int | None = None,
    ) -> list[float]:
        """Train the network on the examples for several epochs.

        At every epoch, the examples are shuffled and cut into mini-batches.
        For every mini-batch: forward pass, loss, backward pass, then the
        optimizer changes the weights.

        Args:
            inputs: Examples, shape (examples, inputs of the first layer).
            targets: Expected outputs, one row per example.
            loss: Measures how wrong the network is.
            optimizer: Changes the weights after each mini-batch. It must
                work on this network.
            epochs: Number of passes over all the examples.
            batch_size: Number of examples in each mini-batch. The last
                mini-batch of an epoch can be smaller.
            seed: Seed of the shuffling, to train the same way every time.

        Returns:
            The average loss of each epoch, from the first to the last.

        Raises:
            ValueError: If epochs or batch_size is not positive, or if
                inputs and targets do not have the same number of examples.
        """
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("Epochs and batch_size must be positive")
        if len(inputs) != len(targets):
            raise ValueError("Inputs and targets must have the same number of examples")

        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(inputs))
            total_loss = 0.0
            for start in range(0, len(inputs), batch_size):
                batch = order[start:start + batch_size]
                predictions = self.forward(inputs[batch])
                batch_loss = loss.forward(predictions, targets[batch])
                self.backward(loss.backward())
                optimizer.step()
                total_loss += batch_loss * len(batch)
            history.append(total_loss / len(inputs))
        return history

    def save(self, path: str) -> None:
        """Write the network to one file, to load it again later.

        The file is a compressed NumPy archive holding:
        - "architecture": JSON text with the file format, its version, the
          description of every layer and the settings.
        - "layer_<i>_param_<j>": the j-th learned array of the i-th layer.

        Args:
            path: Where to write the file, e.g. "my_torch_network.nn".
        """
        architecture = {
            "format": FILE_FORMAT,
            "version": FILE_VERSION,
            "layers": [layer.config() for layer in self.layers],
            "settings": self.settings,
        }
        arrays = {"architecture": np.array(json.dumps(architecture))}
        for layer_index, layer in enumerate(self.layers):
            for param_index, (value, _) in enumerate(layer.parameters()):
                arrays[f"layer_{layer_index}_param_{param_index}"] = value
        with open(path, "wb") as file:
            np.savez_compressed(file, **arrays)
