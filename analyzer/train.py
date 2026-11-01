"""
Train mode
"""

import numpy as np

from analyzer.chess_network import check_chess_network
from analyzer.chessfile import read_chessfile
from analyzer.encoding import encode_boards
from analyzer.labels import LABELS, label_to_index
from my_torch import SGD, CrossEntropyLoss, MSELoss, Network

DEFAULT_SETTINGS = {
    "loss": "CrossEntropyLoss",
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "epochs": 10,
    "batch_size": 64,
    "seed": None,
}
LOSSES = {"CrossEntropyLoss": CrossEntropyLoss, "MSELoss": MSELoss}
OPTIMIZERS = {"SGD": SGD}


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
    return merged


def encode_answers(answers: list[str]) -> np.ndarray:
    """Turn the expected answers into one-hot rows.

    "Check White" becomes [0, 1, 0, 0, 0].

    Returns:
        Array of shape (number of answers, 5).
    """
    indexes = [label_to_index(answer) for answer in answers]
    return np.eye(len(LABELS))[indexes]


def run_train(loadfile: str, chessfile: str) -> None:
    """Train a network on the chessboards of a file, then save it.

    The loss, optimizer, learning rate, number of epochs, batch size and
    seed come from the settings of the network. The network is saved over
    LOADFILE.

    Args:
        loadfile: The network file.
        chessfile: The chessboard file; every line needs its expected answer.

    Raises:
        FileNotFoundError: If a file does not exist.
        ValueError: If the network, its settings or the chessboard file is
            not valid.
    """
    network = Network.load(loadfile)
    check_chess_network(network)
    settings = read_settings(network.settings)
    fens, answers = read_chessfile(chessfile, answers_needed=True)
    if not fens:
        raise ValueError(f"{chessfile}: no chessboard to train on")

    inputs = encode_boards(fens)
    targets = encode_answers(answers)
    loss = LOSSES[settings["loss"]]()
    optimizer = OPTIMIZERS[settings["optimizer"]](network, settings["learning_rate"])
    history = network.train(inputs, targets, loss, optimizer,
                            settings["epochs"], settings["batch_size"], settings["seed"])
    network.save(loadfile)
    print(f"Trained on {len(fens)} chessboards for {settings['epochs']} epochs: "
          f"loss {history[0]:.4f} -> {history[-1]:.4f}")
