"""
Predict mode tests
"""

import pytest

import analyzer.predict
from analyzer.chess_network import check_chess_network
from analyzer.cli import main
from analyzer.encoding import encode_boards
from analyzer.labels import LABELS
from analyzer.predict import predict_answers
from my_torch import Linear, Network, ReLU, Softmax

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHECK = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59"
MATE = "2RRQQ1K/6N1/8/3k4/8/8/8/8 b - - 0 79"


def chess_network():
    return Network([Linear(769, 8, seed=1), ReLU(), Linear(8, 5, seed=2), Softmax()])


@pytest.fixture
def files(tmp_path):
    """A saved chess network and a chessboard file, as paths."""
    network = chess_network()
    network_path = tmp_path / "network.nn"
    network.save(str(network_path))
    boards_path = tmp_path / "boards.txt"
    boards_path.write_text(f"{START}\n{CHECK} Check White\n\n{MATE}\n")
    return network, str(network_path), str(boards_path)


def test_predict_prints_one_answer_per_chessboard(files, capsys):
    network, network_path, boards_path = files

    assert main(["--predict", network_path, boards_path]) == 0

    out = capsys.readouterr().out
    assert out.endswith("\n")
    lines = out.splitlines()
    expected = [LABELS[index] for index in network.predict(encode_boards([START, CHECK, MATE]))]
    assert lines == expected


def test_predict_answers_are_known_labels(files, capsys):
    _, network_path, boards_path = files
    main(["--predict", network_path, boards_path])
    assert set(capsys.readouterr().out.splitlines()) <= set(LABELS)


def test_predict_of_an_empty_file_prints_nothing(files, tmp_path, capsys):
    _, network_path, _ = files
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert main(["--predict", network_path, str(empty)]) == 0
    assert capsys.readouterr().out == ""


def test_predict_answers_are_the_same_in_chunks(monkeypatch):
    network = chess_network()
    fens = [START, CHECK, MATE] * 4
    all_at_once = predict_answers(network, fens)
    monkeypatch.setattr(analyzer.predict, "CHUNK_SIZE", 5)
    assert predict_answers(network, fens) == all_at_once


def test_check_chess_network_accepts_a_chess_network():
    check_chess_network(chess_network())


@pytest.mark.parametrize("layers", [
    [Linear(2, 4), ReLU(), Linear(4, 2), Softmax()],
    [Linear(769, 4), ReLU(), Linear(4, 3), Softmax()],
    [ReLU()],
])
def test_check_chess_network_rejects_other_networks(layers):
    with pytest.raises(ValueError, match="does not take a chessboard"):
        check_chess_network(Network(layers))


def test_predict_with_a_non_chess_network_returns_84(files, tmp_path, capsys):
    _, _, boards_path = files
    xor_path = tmp_path / "xor.nn"
    Network([Linear(2, 2), Softmax()]).save(str(xor_path))
    assert main(["--predict", str(xor_path), boards_path]) == 84
    output = capsys.readouterr()
    assert output.out == ""
    assert "does not take a chessboard" in output.err


def test_predict_with_a_broken_chessboard_file_prints_no_answer(files, tmp_path, capsys):
    _, network_path, _ = files
    broken = tmp_path / "broken.txt"
    broken.write_text(f"{START}\n{CHECK}\nnot a chessboard\n")
    result = main(["--predict", network_path, str(broken)])
    output = capsys.readouterr()
    assert result == 84
    assert output.out == ""
    assert "line 3: invalid FEN" in output.err


def test_predict_with_a_broken_network_file_returns_84(files, capsys):
    _, _, boards_path = files
    assert main(["--predict", boards_path, boards_path]) == 84
    assert "not a my_torch network file" in capsys.readouterr().err


def test_predict_ignores_expected_answers(files, tmp_path, capsys):
    _, network_path, _ = files
    with_answers = tmp_path / "with_answers.txt"
    without_answers = tmp_path / "without_answers.txt"
    with_answers.write_text(f"{CHECK} Check White\n{START} Nothing\n")
    without_answers.write_text(f"{CHECK}\n{START}\n")
    main(["--predict", network_path, str(with_answers)])
    first = capsys.readouterr().out
    main(["--predict", network_path, str(without_answers)])
    assert capsys.readouterr().out == first

