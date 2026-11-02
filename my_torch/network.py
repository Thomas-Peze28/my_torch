"""
Neural network
"""

import json
import zipfile
from typing import TYPE_CHECKING

import numpy as np

from my_torch.activations import ReLU, Sigmoid, Softmax
from my_torch.layers import Layer, Linear
from my_torch.layers.layer import Parameter
from my_torch.losses import Loss

if TYPE_CHECKING:
    from my_torch.optimizers import Optimizer

Examples = tuple[np.ndarray, np.ndarray]
"""Examples and their expected outputs: (inputs, targets), one row per example."""

FILE_FORMAT = "my_torch"
FILE_VERSION = 1.0

LAYER_TYPES = {"Linear": Linear, "ReLU": ReLU, "Sigmoid": Sigmoid, "Softmax": Softmax}


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

    def parameters(self) -> list[Parameter]:
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
        optimizer: "Optimizer",
        epochs: int,
        batch_size: int,
        seed: int | None = None,
        validation: Examples | None = None,
        patience: int | None = None,
        verbose: bool = False,
    ) -> dict:
        """Train the network on the examples for several epochs.

        At every epoch, the examples are shuffled and cut into mini-batches.
        For every mini-batch: forward pass, loss, backward pass, then the
        optimizer changes the weights.

        With validation examples, the network is also checked on them after
        every epoch. It keeps the weights of its best epoch (highest
        validation accuracy), and stops early when the validation accuracy
        has not improved for `patience` epochs: going on would only learn
        the training examples by heart.

        Args:
            inputs: Examples, shape (examples, inputs of the first layer).
            targets: Expected outputs, one row per example.
            loss: Measures how wrong the network is.
            optimizer: Changes the weights after each mini-batch. It must
                work on this network.
            epochs: Maximum number of passes over all the examples.
            batch_size: Number of examples in each mini-batch. The last
                mini-batch of an epoch can be smaller.
            seed: Seed of the shuffling, to train the same way every time.
            validation: (inputs, targets) never used to learn, only to check
                the network. None trains without checking.
            patience: Number of epochs without progress on the validation
                before stopping. None never stops early.
            verbose: True prints one line per epoch.

        Returns:
            The history of the training, one value per epoch:
            {"loss": [...], "accuracy": [...], "val_loss": [...],
            "val_accuracy": [...], "best_epoch": n}. Accuracy is the share
            of examples whose biggest output is the expected one. The
            validation lists are empty and best_epoch is None without
            validation.

        Raises:
            ValueError: If epochs, batch_size or patience is not positive,
                or if inputs and targets do not have the same number of
                examples.
        """
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("Epochs and batch_size must be positive")
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be positive")
        if len(inputs) != len(targets):
            raise ValueError("Inputs and targets must have the same number of examples")

        rng = np.random.default_rng(seed)
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "best_epoch": None}
        best_weights = None
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self._train_one_epoch(inputs, targets, loss, optimizer, batch_size, rng)
            history["loss"].append(train_loss)
            history["accuracy"].append(train_accuracy)
            message = f"Epoch {epoch}/{epochs}: loss {train_loss:.4f}, accuracy {train_accuracy:.1%}"

            if validation is not None:
                val_loss, val_accuracy = self.evaluate(validation[0], validation[1], loss)
                is_best = val_accuracy > max(history["val_accuracy"], default=-1)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
                message += f" | validation loss {val_loss:.4f}, accuracy {val_accuracy:.1%}"
                if is_best:
                    history["best_epoch"] = epoch
                    best_weights = self._copy_weights()
                    message += " (best)"

            if verbose:
                print(message, flush=True)
            if patience is not None and history["best_epoch"] is not None \
                    and epoch - history["best_epoch"] >= patience:
                if verbose:
                    print(f"No progress for {patience} epochs: stopped", flush=True)
                break

        if best_weights is not None:
            self._set_weights(best_weights)
        return history

    def _train_one_epoch(self, inputs: np.ndarray, targets: np.ndarray, loss: Loss, optimizer: "Optimizer",
                         batch_size: int, rng: np.random.Generator) -> tuple[float, float]:
        """Learn once from every example, in mini-batches of a random order.

        Returns:
            (average loss, accuracy) over the examples of this epoch,
            measured on the mini-batches while learning.
        """
        order = rng.permutation(len(inputs))
        total_loss = 0.0
        right = 0
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            predictions = self.forward(inputs[batch])
            total_loss += loss.forward(predictions, targets[batch]) * len(batch)
            right += np.sum(predictions.argmax(axis=1) == targets[batch].argmax(axis=1))
            self.backward(loss.backward())
            optimizer.step()
        return float(total_loss / len(inputs)), float(right / len(inputs))

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, loss: Loss) -> tuple[float, float]:
        """Check the network on examples, without learning from them.

        Returns:
            (loss, accuracy). Accuracy is the share of examples whose
            biggest output is the expected one (0.9 = 90%).
        """
        predictions = self.forward(inputs)
        accuracy = np.mean(predictions.argmax(axis=1) == targets.argmax(axis=1))
        return float(loss.forward(predictions, targets)), float(accuracy)

    def _copy_weights(self) -> list[np.ndarray]:
        """Take a copy of every weight and bias, like a photo of the network."""
        return [value.copy() for value, _ in self.parameters()]

    def _set_weights(self, copies: list[np.ndarray]) -> None:
        """Put back the weights and biases of a copy made by _copy_weights()."""
        for (value, _), copy in zip(self.parameters(), copies):
            value[...] = copy

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

    @classmethod
    def load(cls, path: str) -> "Network":
        """Build a network from a file written by save().

        Args:
            path: The file to read.

        Returns:
            A new network with the same layers, learned arrays and settings
            as the saved one.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid my_torch network file.
        """
        try:
            archive = np.load(path, allow_pickle=False)
        except FileNotFoundError:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            raise ValueError(f"{path}: not a my_torch network file") from None
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"{path}: not a my_torch network file")
        with archive:
            try:
                return cls._from_archive(archive)
            except (KeyError, TypeError, AttributeError, ValueError) as error:
                raise ValueError(f"{path}: not a my_torch network file ({error})") from None

    @classmethod
    def _from_archive(cls, archive: np.lib.npyio.NpzFile) -> "Network":
        """Build a network from an opened archive.

        Raises:
            ValueError: If the architecture or an array is not valid.
            KeyError: If the architecture or an array is missing.
        """
        if "architecture" not in archive.files:
            raise ValueError("No architecture")
        architecture = json.loads(str(archive["architecture"]))
        if architecture["format"] != FILE_FORMAT or architecture["version"] != FILE_VERSION:
            raise ValueError("Unknown format or version")
        network = cls.from_config(architecture)
        for layer_index, layer in enumerate(network.layers):
            for param_index, (value, _) in enumerate(layer.parameters()):
                saved = archive[f"layer_{layer_index}_param_{param_index}"]
                if saved.shape != value.shape:
                    raise ValueError(f"Wrong shape for layer {layer_index}")
                value[...] = saved
        return network

    @classmethod
    def from_config(cls, description: dict, seed: int | None = None) -> "Network":
        """Build a new network, with random weights, from its description.

        The description is the one written in configuration files and in
        saved networks:
        ```
            {
                "layers": [{"type": "Linear", "input_size": 769, "output_size": 256},
                           {"type": "ReLU"}, ...],
                "settings": {"learning_rate": 0.1, ...}
            }
        "settings" #is optional. Other keys are ignored.
        ```

        Args:
            description: The layers, in order, and the optional settings.
            seed: Seed of the random starting weights, to build the same
                network every time. None gives different weights each time.

        Returns:
            A new, untrained network.

        Raises:
            ValueError: If the description is not valid: no layers, unknown
                layer type, wrong layer settings, or sizes that do not
                follow each other.
        """
        if not isinstance(description, dict):
            raise ValueError("A network description must be a JSON object")
        configs = description.get("layers")
        if not isinstance(configs, list) or not configs:
            raise ValueError('A network description needs a non-empty "layers" list')
        settings = description.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError('"settings" must be a JSON object')

        rng = np.random.default_rng(seed)
        layers = []
        size = None
        for layer_index, config in enumerate(configs):
            if not isinstance(config, dict) or "type" not in config:
                raise ValueError(f'Layer {layer_index} needs a "type"')
            arguments = dict(config)
            layer_type = arguments.pop("type")
            if layer_type not in LAYER_TYPES:
                raise ValueError(f'Unknown layer type "{layer_type}"')
            if layer_type == "Linear":
                arguments["seed"] = int(rng.integers(2**32))
            try:
                layer = LAYER_TYPES[layer_type](**arguments)
            except TypeError:
                raise ValueError(f"Wrong settings for layer {layer_index}: {config}") from None
            if layer_type == "Linear":
                if size is not None and config["input_size"] != size:
                    raise ValueError(f"Layer {layer_index} gets {config['input_size']} inputs, "
                                     f"but the layer before gives {size}")
                size = config["output_size"]
            layers.append(layer)
        return cls(layers, settings=settings)
