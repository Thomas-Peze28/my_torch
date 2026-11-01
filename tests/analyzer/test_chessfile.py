"""
Chessboard file reading tests
"""

import pytest

from analyzer.chessfile import read_chessfile

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHECK = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59"


def write(tmp_path, text, name="boards.txt"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_read_chessfile_gives_fens_and_answers_in_order(tmp_path):
    path = write(tmp_path, f"{CHECK} Check White\n{START} Nothing\n")
    fens, answers = read_chessfile(path, answers_needed=True)
    assert fens == [CHECK, START]
    assert answers == ["Check White", "Nothing"]


def test_read_chessfile_without_answers_in_predict_mode(tmp_path):
    path = write(tmp_path, f"{START}\n{CHECK}\n")
    fens, answers = read_chessfile(path, answers_needed=False)
    assert fens == [START, CHECK]
    assert answers == ["", ""]


def test_read_chessfile_allows_answers_in_predict_mode(tmp_path):
    path = write(tmp_path, f"{CHECK} Check White\n{START}\n")
    fens, answers = read_chessfile(path, answers_needed=False)
    assert fens == [CHECK, START]
    assert answers == ["Check White", ""]


def test_read_chessfile_ignores_unknown_answers_in_predict_mode(tmp_path):
    path = write(tmp_path, f"{CHECK} Chek White\n")
    fens, _ = read_chessfile(path, answers_needed=False)
    assert fens == [CHECK]


def test_read_chessfile_skips_empty_lines(tmp_path):
    path = write(tmp_path, f"\n{START}\n   \n{CHECK}\n\n")
    fens, _ = read_chessfile(path, answers_needed=False)
    assert fens == [START, CHECK]


def test_read_chessfile_accepts_a_last_line_without_newline(tmp_path):
    path = write(tmp_path, f"{START} Nothing")
    assert read_chessfile(path, answers_needed=True) == ([START], ["Nothing"])


def test_read_chessfile_of_an_empty_file_gives_nothing(tmp_path):
    assert read_chessfile(write(tmp_path, ""), answers_needed=False) == ([], [])


def test_read_chessfile_reads_the_dataset_format(tmp_path):
    lines = ("8/8/8/1p1p2p1/q1p5/8/K1k5/8 w - - 4 70 Checkmate Black\n"
             "2RRQQ1K/6N1/8/3k4/8/8/8/8 b - - 0 79 Checkmate White\n"
             "8/5p1p/5K1P/6P1/7k/8/8/6R1 b - - 14 80 Nothing\n")
    fens, answers = read_chessfile(write(tmp_path, lines), answers_needed=True)
    assert fens[1] == "2RRQQ1K/6N1/8/3k4/8/8/8/8 b - - 0 79"
    assert answers == ["Checkmate Black", "Checkmate White", "Nothing"]


@pytest.mark.parametrize("text, answers_needed, message", [
    (f"{START}\nnot a chessboard\n", False, "line 2: invalid FEN"),
    (f"{START} Nothing\n{CHECK}\n", True, "line 2: missing expected answer"),
    (f"{CHECK} Chek White\n", True, 'line 1: unknown answer "Chek White"'),
    (f"{CHECK} Check\n", True, 'line 1: unknown answer "Check"'),
])
def test_read_chessfile_reports_the_line_of_an_error(tmp_path, text, answers_needed, message):
    with pytest.raises(ValueError, match=message):
        read_chessfile(write(tmp_path, text), answers_needed)


def test_read_chessfile_of_a_binary_file_raises_value_error(tmp_path):
    path = tmp_path / "network.nn"
    path.write_bytes(bytes(range(256)))
    with pytest.raises(ValueError, match="not a text file"):
        read_chessfile(str(path), answers_needed=False)


def test_read_chessfile_of_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_chessfile(str(tmp_path / "missing.txt"), answers_needed=False)
