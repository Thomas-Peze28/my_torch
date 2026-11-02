"""
Train mode
"""

import os

import numpy as np

from analyzer.chess_network import check_chess_network
from analyzer.chessfile import read_chessfile
from analyzer.encoding import encode_boards
from analyzer.labels import LABELS, label_to_index
from my_torch import SGD, Adam, CrossEntropyLoss, MSELoss, Network
from my_torch.network import Examples

DEFAULT_SETTINGS = {
    "loss": "CrossEntropyLoss",
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "epochs": 10,
    "batch_size": 64,
    "seed": None,
    "validation_split": 0.1,
    "patience": 3,
}
LOSSES = {"CrossEntropyLoss": CrossEntropyLoss, "MSELoss": MSELoss}
OPTIMIZERS = {"SGD": SGD, "Adam": Adam}


def _is_number(value: object) -> bool:
    """Return True if the value is a number, but not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_settings(settings: dict) -> dict:
    """Complete the training settings of a network with the default ones.

    Keys that are not in DEFAULT_SETTINGS are kept but not used.

    Args:
        settings: The settings saved in the network.

    Returns:
        The settings, with every key of DEFAULT_SETTINGS.

    Raises:
        ValueError: If a setting is not valid.
    """
    merged = DEFAULT_SETTINGS | settings
    if merged["loss"] not in LOSSES:
        raise ValueError(f'unknown loss "{merged["loss"]}" in the network settings')
    if merged["optimizer"] not in OPTIMIZERS:
        raise ValueError(f'unknown optimizer "{merged["optimizer"]}" in the network settings')
    if not _is_number(merged["learning_rate"]) or merged["learning_rate"] <= 0:
        raise ValueError('"learning_rate" must be a positive number in the network settings')
    for key in ("epochs", "batch_size"):
        if not isinstance(merged[key], int) or isinstance(merged[key], bool) or merged[key] <= 0:
            raise ValueError(f'"{key}" must be a positive whole number in the network settings')
    if merged["seed"] is not None and (not isinstance(merged["seed"], int) or isinstance(merged["seed"], bool)):
        raise ValueError('"seed" must be a whole number in the network settings')
    if not _is_number(merged["validation_split"]) or not 0 <= merged["validation_split"] < 1:
        raise ValueError('"validation_split" must be a number from 0 to 1 (excluded) in the network settings')
    if not isinstance(merged["patience"], int) or isinstance(merged["patience"], bool) or merged["patience"] <= 0:
        raise ValueError('"patience" must be a positive whole number in the network settings')
    return merged


def encode_answers(answers: list[str]) -> np.ndarray:
    """Turn the expected answers into one-hot rows.

    "Check White" becomes [0, 1, 0, 0, 0].

    Returns:
        Array of shape (number of answers, 5).
    """
    indexes = [label_to_index(answer) for answer in answers]
    return np.eye(len(LABELS))[indexes]


def split_validation(inputs: np.ndarray, targets: np.ndarray, share: float,
                     seed: int | None) -> tuple[Examples, Examples | None]:
    """Put a random part of the chessboards aside to check the network.

    The chessboards put aside are never used to learn: they show if the
    network understands, or only learns the other ones by heart.

    Args:
        inputs: Encoded chessboards, one row each.
        targets: Expected answers, one one-hot row each.
        share: Part put aside (0.1 = 10%). 0 puts nothing aside.
        seed: Seed of the random choice.

    Returns:
        ((train inputs, train targets), (validation inputs, validation
        targets)), or None for the validation if too few chessboards are
        put aside (less than 1).
    """
    size = int(len(inputs) * share)
    if size == 0:
        return (inputs, targets), None
    order = np.random.default_rng(seed).permutation(len(inputs))
    kept, aside = order[size:], order[:size]
    return (inputs[kept], targets[kept]), (inputs[aside], targets[aside])


def check_writable(path: str) -> None:
    """Make sure a network can be written to a path, before training.

    Raises:
        ValueError: If the folder does not exist, or the file or the folder
            cannot be written.
    """
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        raise ValueError(f"cannot save the network to {path}: the folder does not exist")
    if os.path.isdir(path):
        raise ValueError(f"cannot save the network to {path}: it is a folder")
    target = path if os.path.exists(path) else folder
    if not os.access(target, os.W_OK):
        raise ValueError(f"cannot save the network to {path}: permission denied")


def run_train(loadfile: str, chessfile: str, savefile: str | None = None) -> None:
    """Train a network on the chessboards of a file, then save it.

    The loss, optimizer, learning rate, number of epochs, batch size and
    seed come from the settings of the network. A part of the chessboards
    ("validation_split") is put aside to check the network after every
    epoch: the epoch with the most right answers on them is kept, and the
    training stops when there is no progress for "patience" epochs. Everything, including where to save,
    is checked before training, so a long training is never lost.

    Args:
        loadfile: The network file.
        chessfile: The chessboard file; every line needs its expected answer.
        savefile: Where to save the trained network. None saves it over
            loadfile.

    Raises:
        FileNotFoundError: If a file does not exist.
        ValueError: If the network, its settings or the chessboard file is
            not valid, or the network cannot be saved.
    """
    target = savefile if savefile is not None else loadfile
    network = Network.load(loadfile)
    check_chess_network(network)
    settings = read_settings(network.settings)
    fens, answers = read_chessfile(chessfile, answers_needed=True)
    if not fens:
        raise ValueError(f"{chessfile}: no chessboard to train on")
    check_writable(target)

    (inputs, targets), validation = split_validation(encode_boards(fens), encode_answers(answers),
                                                     settings["validation_split"], settings["seed"])
    loss = LOSSES[settings["loss"]]()
    optimizer = OPTIMIZERS[settings["optimizer"]](network, settings["learning_rate"])
    history = network.train(inputs, targets, loss, optimizer, settings["epochs"], settings["batch_size"],
                            settings["seed"], validation, settings["patience"], verbose=True)
    network.save(target)
    print(f"Trained on {len(inputs)} chessboards for {len(history['loss'])} epochs: "
          f"loss {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")
    if validation is not None:
        best = history["best_epoch"]
        print(f"Kept epoch {best}, checked on {len(validation[0])} chessboards put aside: "
              f"validation accuracy {history['val_accuracy'][best - 1]:.1%}")
