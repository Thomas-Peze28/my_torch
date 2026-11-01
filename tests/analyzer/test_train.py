"""
Train mode tests
"""

import numpy as np
import pytest

from analyzer.cli import main
from analyzer.encoding import encode_boards
from analyzer.labels import LABELS
from analyzer.train import DEFAULT_SETTINGS, encode_answers, read_settings
from my_torch import Linear, Network, ReLU, Softmax

BOARDS = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "Nothing",
    "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59": "Check White",
    "2r2b1k/5Pp1/4q2p/2pbP2P/3p2R1/5N1Q/3BnP2/1q3NK1 w - - 2 41": "Check Black",
    "2RRQQ1K/6N1/8/3k4/8/8/8/8 b - - 0 79": "Checkmate White",
    "8/8/8/1p1p2p1/q1p5/8/K1k5/8 w - - 4 70": "Checkmate Black",
}
SETTINGS = {"loss": "CrossEntropyLoss", "optimizer": "SGD", "learning_rate": 0.05,
            "epochs": 300, "batch_size": 5, "seed": 0}


def save_network(path, settings=SETTINGS, inputs=769):
    Network([Linear(inputs, 16, seed=1), ReLU(), Linear(16, 5, seed=2), Softmax()], settings).save(str(path))
    return str(path)


def write_boards(path, boards=BOARDS):
    path.write_text("".join(f"{fen} {answer}\n" for fen, answer in boards.items()))
    return str(path)


@pytest.fixture
def files(tmp_path):
    return save_network(tmp_path / "network.nn"), write_boards(tmp_path / "boards.txt")


def test_read_settings_fills_the_missing_settings():
    assert read_settings({}) == DEFAULT_SETTINGS
    assert read_settings({"epochs": 3})["epochs"] == 3
    assert read_settings({"comment": "kept"})["comment"] == "kept"


@pytest.mark.parametrize("settings, message", [
    ({"loss": "Keras"}, "unknown loss"),
    ({"optimizer": "Adam"}, "unknown optimizer"),
    ({"learning_rate": 0}, "learning_rate"),
    ({"learning_rate": "fast"}, "learning_rate"),
    ({"epochs": -3}, "epochs"),
    ({"epochs": 2.5}, "epochs"),
    ({"batch_size": True}, "batch_size"),
    ({"seed": "abc"}, "seed"),
])
def test_read_settings_rejects_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        read_settings(settings)


def test_encode_answers_gives_one_hot_rows():
    np.testing.assert_array_equal(encode_answers(["Check White", "Nothing"]),
                                  [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])


def test_train_learns_the_chessboards_of_the_file(files, capsys):
    network_path, boards_path = files

    assert main(["--train", network_path, boards_path]) == 0

    assert "Trained on 5 chessboards for 300 epochs" in capsys.readouterr().out
    trained = Network.load(network_path)
    answers = [LABELS[index] for index in trained.predict(encode_boards(list(BOARDS)))]
    assert answers == list(BOARDS.values())


def test_train_keeps_the_settings_of_the_network(files):
    network_path, boards_path = files
    main(["--train", network_path, boards_path])
    assert Network.load(network_path).settings == SETTINGS


def test_train_with_the_same_seed_gives_the_same_network(tmp_path):
    boards_path = write_boards(tmp_path / "boards.txt")
    first = save_network(tmp_path / "first.nn")
    second = save_network(tmp_path / "second.nn")
    main(["--train", first, boards_path])
    main(["--train", second, boards_path])
    for (first_value, _), (second_value, _) in zip(Network.load(first).parameters(),
                                                     Network.load(second).parameters()):
        np.testing.assert_array_equal(first_value, second_value)


def test_train_uses_the_default_settings_when_there_are_none(tmp_path):
    network_path = save_network(tmp_path / "network.nn", settings={})
    assert main(["--train", network_path, write_boards(tmp_path / "boards.txt")]) == 0


@pytest.mark.parametrize("case", ["missing answer", "unknown answer", "empty file",
                                  "bad settings", "not a chess network"])
def test_train_errors_return_84_and_keep_the_network(tmp_path, capsys, case):
    network_path = save_network(tmp_path / "network.nn",
                                settings={"loss": "Keras"} if case == "bad settings" else SETTINGS,
                                inputs=2 if case == "not a chess network" else 769)
    boards = tmp_path / "boards.txt"
    lines = {
        "missing answer": "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59\n",
        "unknown answer": "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Chek White\n",
        "empty file": "",
    }
    if case in lines:
        boards.write_text(lines[case])
    else:
        write_boards(boards)
    before = (tmp_path / "network.nn").read_bytes()

    assert main(["--train", network_path, str(boards)]) == 84

    output = capsys.readouterr()
    assert output.err.startswith("my_torch_analyzer: ")
    assert output.out == ""
    assert (tmp_path / "network.nn").read_bytes() == before
