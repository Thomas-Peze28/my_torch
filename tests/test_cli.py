"""
CLI tests
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "my_torch_analyzer"


def run(*args):
    """Run my_torch_analyzer like the grader would, and return the result."""
    return subprocess.run(
        [sys.executable, str(PROGRAM), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.fixture
def files(tmp_path):
    """Create a fake network file and a fake chessboard file, so the paths exist."""
    network = tmp_path / "network.nn"
    boards = tmp_path / "boards.txt"
    network.write_text("")
    boards.write_text("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n")
    return str(network), str(boards)


def test_help_prints_usage_and_returns_0():
    result = run("--help")
    assert result.returncode == 0
    assert "USAGE" in result.stdout


@pytest.mark.parametrize("args", [
    [],                                          # nothing at all
    ["--predict"],                               # files are missing
    ["--predict", "only_one_file"],              # one file is missing
    ["--train", "--predict", "a", "b"],          # two modes at once
    ["--predict", "--save", "x", "a", "b"],      # --save without --train
    ["--train", "--save"],                       # --save without a file name
    ["--unknown", "a", "b"],                     # unknown option
    ["--predict", "missing.nn", "missing.txt"],  # files do not exist
])
def test_invalid_arguments_return_84(args):
    result = run(*args)
    assert result.returncode == 84
    assert result.stderr != ""   # the error message goes to the error output
    assert result.stdout == ""   # and nothing goes to the normal output


def test_valid_arguments_are_accepted(files):
    from analyzer.cli import parse_args

    network, boards = files
    options = parse_args(["--train", "--save", "new.nn", network, boards])
    assert options.mode == "train"
    assert options.savefile == "new.nn"
    assert options.loadfile == network
    assert options.chessfile == boards

    options = parse_args(["--predict", network, boards])
    assert options.mode == "predict"
    assert options.savefile is None
