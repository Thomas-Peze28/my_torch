"""
Analyzer error tests

The real program is run on broken files, like the grader would: it must
print one message on the error output and exit with 84, never crash.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from my_torch import Linear, Network, ReLU, Softmax

PROGRAM = Path(__file__).resolve().parent.parent.parent / "my_torch_analyzer"
CHECK = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59"


@pytest.fixture
def folder(tmp_path):
    """A folder with a valid chess network, a non-chess network, and broken files."""
    Network([Linear(769, 4, seed=1), ReLU(), Linear(4, 5, seed=2), Softmax()]).save(str(tmp_path / "chess.nn"))
    Network([Linear(2, 2, seed=1), Softmax()]).save(str(tmp_path / "xor.nn"))
    (tmp_path / "cut.nn").write_bytes((tmp_path / "chess.nn").read_bytes()[:500])
    (tmp_path / "empty.nn").write_bytes(b"")
    (tmp_path / "ok.txt").write_text(f"{CHECK} Check White\n")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00broken")
    (tmp_path / "bad_fen.txt").write_text(f"{CHECK} Check White\nrnbqkbnr/ppp w\n")
    (tmp_path / "no_answer.txt").write_text(f"{CHECK}\n")
    (tmp_path / "bad_answer.txt").write_text(f"{CHECK} Check Purple\n")
    (tmp_path / "empty.txt").write_text("")
    return tmp_path


def run(folder, *args):
    return subprocess.run([sys.executable, str(PROGRAM), *args],
                          capture_output=True, text=True, cwd=folder)


@pytest.mark.parametrize("args, message", [
    (["--predict", "empty.nn", "ok.txt"], "not a my_torch network file"),
    (["--predict", "cut.nn", "ok.txt"], "not a my_torch network file"),
    (["--predict", "ok.txt", "ok.txt"], "not a my_torch network file"),
    (["--predict", "xor.nn", "ok.txt"], "does not take a chessboard"),
    (["--predict", "chess.nn", "binary.txt"], "not a text file"),
    (["--predict", "chess.nn", "bad_fen.txt"], "line 2: invalid FEN"),
    (["--train", "empty.nn", "ok.txt"], "not a my_torch network file"),
    (["--train", "xor.nn", "ok.txt"], "does not take a chessboard"),
    (["--train", "chess.nn", "binary.txt"], "not a text file"),
    (["--train", "chess.nn", "bad_fen.txt"], "line 2: invalid FEN"),
    (["--train", "chess.nn", "no_answer.txt"], "line 1: missing expected answer"),
    (["--train", "chess.nn", "bad_answer.txt"], 'line 1: unknown answer "Check Purple"'),
    (["--train", "chess.nn", "empty.txt"], "no chessboard to train on"),
    (["--train", "--save", "missing/new.nn", "chess.nn", "ok.txt"], "folder does not exist"),
])
def test_broken_files_exit_with_84_and_one_message(folder, args, message):
    result = run(folder, *args)
    assert result.returncode == 84
    assert result.stdout == ""
    assert result.stderr.startswith("my_torch_analyzer: ")
    assert message in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root can read any file")
def test_unreadable_chessboard_file_exits_with_84(folder):
    locked = folder / "locked.txt"
    locked.write_text(f"{CHECK}\n")
    locked.chmod(0)
    try:
        result = run(folder, "--predict", "chess.nn", "locked.txt")
    finally:
        locked.chmod(0o600)
    assert result.returncode == 84
    assert "Permission denied" in result.stderr
    assert "Traceback" not in result.stderr


def test_failed_training_keeps_the_network_file(folder):
    before = (folder / "chess.nn").read_bytes()
    run(folder, "--train", "chess.nn", "bad_answer.txt")
    assert (folder / "chess.nn").read_bytes() == before


def test_unexpected_errors_also_exit_with_84(folder, monkeypatch, capsys):
    import analyzer.cli

    def broken(*args):
        raise RuntimeError("something nobody planned")

    monkeypatch.setattr(analyzer.cli, "run_predict", broken)
    assert analyzer.cli.main(["--predict", str(folder / "chess.nn"), str(folder / "ok.txt")]) == 84
    assert "unexpected error" in capsys.readouterr().err
