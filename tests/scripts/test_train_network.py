"""
Network training script tests

They train a tiny network on a tiny fake dataset, because the real one is
not in the repository.
"""

import json

import pytest

from my_torch import Linear, Network, Softmax
from scripts.train_network import accuracy, main

BOARDS = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "Nothing",
    "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59": "Check White",
    "2r2b1k/5Pp1/4q2p/2pbP2P/3p2R1/5N1Q/3BnP2/1q3NK1 w - - 2 41": "Check Black",
    "2RRQQ1K/6N1/8/3k4/8/8/8/8 b - - 0 79": "Checkmate White",
    "8/8/8/1p1p2p1/q1p5/8/K1k5/8 w - - 4 70": "Checkmate Black",
}
CONFIG = {
    "layers": [
        {"type": "Linear", "input_size": 769, "output_size": 16},
        {"type": "ReLU"},
        {"type": "Linear", "input_size": 16, "output_size": 5},
        {"type": "Softmax"},
    ],
    "settings": {"loss": "CrossEntropyLoss", "optimizer": "SGD", "learning_rate": 0.05,
                 "epochs": 300, "batch_size": 5, "seed": 0},
}


@pytest.fixture
def folder(tmp_path):
    """A configuration file and a dataset folder with the three files."""
    (tmp_path / "small.conf").write_text(json.dumps(CONFIG))
    lines = "".join(f"{fen} {answer}\n" for fen, answer in BOARDS.items())
    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.txt").write_text(lines)
    return tmp_path


def run(folder, *extra):
    return main(["--config", str(folder / "small.conf"), "--dataset", str(folder),
                 "--output", str(folder / "my_torch_network.nn"), *extra])


def test_accuracy_counts_the_right_answers(tmp_path):
    network = Network([Linear(769, 5, seed=1), Softmax()])
    network.layers[0].weights[...] = 0
    network.layers[0].bias[...] = [1, 0, 0, 0, 0]
    (tmp_path / "boards.txt").write_text("".join(f"{fen} {answer}\n" for fen, answer in BOARDS.items()))
    assert accuracy(network, str(tmp_path / "boards.txt")) == pytest.approx(1 / 5)


def test_accuracy_fails_on_an_empty_file(tmp_path):
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(ValueError, match="no chessboard to check"):
        accuracy(Network([Linear(769, 5, seed=1), Softmax()]), str(tmp_path / "empty.txt"))


def test_main_creates_and_trains_a_network_that_learns(folder, capsys):
    assert run(folder) == 0
    output = capsys.readouterr().out
    assert "Trained on 5 chessboards for 300 epochs" in output
    assert "Validation accuracy: 100.00%" in output
    assert "Test accuracy" not in output
    assert accuracy(Network.load(str(folder / "my_torch_network.nn")), str(folder / "test.txt")) == 1


def test_main_gives_the_test_accuracy_only_when_asked(folder, capsys):
    assert run(folder, "--test") == 0
    assert "Test accuracy: 100.00%" in capsys.readouterr().out


def test_main_gives_the_same_network_every_time(folder):
    run(folder)
    first = (folder / "my_torch_network.nn").read_bytes()
    run(folder)
    assert (folder / "my_torch_network.nn").read_bytes() == first


@pytest.mark.parametrize("broken", ["missing config", "missing dataset"])
def test_main_returns_84_with_a_message(folder, capsys, broken):
    if broken == "missing config":
        (folder / "small.conf").unlink()
    else:
        (folder / "train.txt").unlink()
    assert run(folder) == 84
    assert capsys.readouterr().err.startswith("train_network: ")
