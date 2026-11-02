"""
Learning curves and evaluation scripts tests
"""

import json

import numpy as np
import pytest

from analyzer.labels import LABELS
from my_torch import Linear, Network, Softmax
from scripts.evaluate_network import answer_accuracy, confusion_matrix
from scripts.evaluate_network import main as evaluate_main
from scripts.plot_history import main as plot_main
from scripts.plot_history import name_of, read_history

PNG_START = b"\x89PNG"
HISTORY = {"loss": [1.0, 0.6, 0.4], "accuracy": [0.5, 0.7, 0.8],
           "val_loss": [1.1, 0.8, 0.9], "val_accuracy": [0.45, 0.65, 0.6], "best_epoch": 2}


def write_history(path, history=HISTORY):
    path.write_text(json.dumps({"config": "small.conf", "history": history, "validation_accuracy": 0.65}))
    return str(path)


def test_read_history_gives_the_history_of_the_file(tmp_path):
    assert read_history(write_history(tmp_path / "run.json")) == HISTORY


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"history": 3}', '{"history": {"loss": []}}'])
def test_read_history_refuses_other_files(tmp_path, content):
    (tmp_path / "bad.json").write_text(content)
    with pytest.raises(ValueError, match="not a training history file"):
        read_history(str(tmp_path / "bad.json"))


def test_name_of_is_the_file_name_without_its_extension():
    assert name_of("benchmarks/init_he.json") == "init_he"


def test_plot_history_draws_one_or_several_trainings(tmp_path):
    first = write_history(tmp_path / "first.json")
    second = write_history(tmp_path / "second.json", {"loss": [0.9, 0.5], "accuracy": [0.6, 0.8],
                                                      "val_loss": [], "val_accuracy": [], "best_epoch": None})
    for paths, output in (([first], "one.png"), ([first, second], "two.png")):
        assert plot_main([*paths, "--output", str(tmp_path / output), "--title", "Test"]) == 0
        assert (tmp_path / output).read_bytes().startswith(PNG_START)


def test_plot_history_returns_84_on_a_broken_file(tmp_path, capsys):
    (tmp_path / "bad.json").write_text("{}")
    assert plot_main([str(tmp_path / "bad.json"), "--output", str(tmp_path / "out.png")]) == 84
    assert capsys.readouterr().err.startswith("plot_history: ")


def test_confusion_matrix_counts_each_expected_and_given_answer():
    expected = ["Nothing", "Nothing", "Check White", "Checkmate Black"]
    answered = ["Nothing", "Check White", "Check White", "Check Black"]
    matrix = confusion_matrix(expected, answered)
    assert matrix.sum() == 4
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1 and matrix[1, 1] == 1 and matrix[4, 2] == 1


def test_answer_accuracy_is_the_share_found_for_each_answer():
    matrix = confusion_matrix(["Nothing", "Nothing", "Check White"], ["Nothing", "Check White", "Check White"])
    np.testing.assert_allclose(answer_accuracy(matrix), [0.5, 1.0, 0.0, 0.0, 0.0])


def test_evaluate_network_prints_the_table_and_draws_the_matrix(tmp_path, capsys):
    network = Network([Linear(769, 5, seed=1), Softmax()])
    network.layers[0].weights[...] = 0
    network.layers[0].bias[...] = [1, 0, 0, 0, 0]
    network.save(str(tmp_path / "network.nn"))
    (tmp_path / "boards.txt").write_text("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing\n"
                                         "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White\n")

    assert evaluate_main([str(tmp_path / "network.nn"), "--file", str(tmp_path / "boards.txt"),
                          "--output", str(tmp_path / "confusion.png")]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["Nothing", "1", "100.0%"]
    assert lines[2].split() == ["Check", "White", "1", "0.0%"]
    assert lines[len(LABELS) + 1].split() == ["all", "2", "50.0%"]
    assert (tmp_path / "confusion.png").read_bytes().startswith(PNG_START)


def test_evaluate_network_returns_84_on_a_missing_network(tmp_path, capsys):
    assert evaluate_main([str(tmp_path / "missing.nn")]) == 84
    assert capsys.readouterr().err.startswith("evaluate_network: ")
