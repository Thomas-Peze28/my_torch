"""
End-to-end tests: the real programs, one after the other

my_torch_generator creates a blank network, my_torch_analyzer --train
teaches it a few chessboards, then my_torch_analyzer --predict answers.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analyzer.encoding import encode_boards
from analyzer.labels import LABELS
from my_torch import Network

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYZER = ROOT / "my_torch_analyzer"
GENERATOR = ROOT / "my_torch_generator"

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


def run(program, folder, *args):
    return subprocess.run([sys.executable, str(program), *args],
                          capture_output=True, text=True, cwd=folder)


@pytest.fixture
def folder(tmp_path):
    """A folder with a configuration file, a training file and a prediction file."""
    (tmp_path / "small.conf").write_text(json.dumps(CONFIG))
    (tmp_path / "train.txt").write_text("".join(f"{fen} {answer}\n" for fen, answer in BOARDS.items()))
    (tmp_path / "boards.txt").write_text("".join(f"{fen}\n" for fen in BOARDS))
    return tmp_path


def test_generate_train_then_predict(folder):
    generated = run(GENERATOR, folder, "small.conf", "1")
    assert generated.returncode == 0

    trained = run(ANALYZER, folder, "--train", "--save", "my_torch_network.nn", "small_1.nn", "train.txt")
    assert trained.returncode == 0, trained.stderr
    assert "Trained on 5 chessboards for 300 epochs" in trained.stdout

    predicted = run(ANALYZER, folder, "--predict", "my_torch_network.nn", "boards.txt")
    assert predicted.returncode == 0, predicted.stderr
    assert predicted.stderr == ""
    assert predicted.stdout.splitlines() == list(BOARDS.values())


def test_predict_before_training_gives_one_known_answer_per_board(folder):
    run(GENERATOR, folder, "small.conf", "1")
    predicted = run(ANALYZER, folder, "--predict", "small_1.nn", "boards.txt")
    lines = predicted.stdout.splitlines()
    assert predicted.returncode == 0
    assert len(lines) == len(BOARDS)
    assert set(lines) <= set(LABELS)


def test_predict_output_is_what_the_network_answers(folder):
    run(GENERATOR, folder, "small.conf", "1")
    network = Network.load(str(folder / "small_1.nn"))
    expected = [LABELS[index] for index in network.predict(encode_boards(list(BOARDS)))]
    predicted = run(ANALYZER, folder, "--predict", "small_1.nn", "boards.txt")
    assert predicted.stdout.splitlines() == expected


def test_train_without_save_updates_the_generated_network(folder):
    run(GENERATOR, folder, "small.conf", "1")
    before = (folder / "small_1.nn").read_bytes()
    assert run(ANALYZER, folder, "--train", "small_1.nn", "train.txt").returncode == 0
    assert (folder / "small_1.nn").read_bytes() != before
    predicted = run(ANALYZER, folder, "--predict", "small_1.nn", "boards.txt")
    assert predicted.stdout.splitlines() == list(BOARDS.values())


def test_the_whole_chain_is_repeatable(folder, tmp_path_factory):
    other = tmp_path_factory.mktemp("again")
    for name in ("small.conf", "train.txt"):
        (other / name).write_bytes((folder / name).read_bytes())
    for place in (folder, other):
        run(GENERATOR, place, "small.conf", "1")
        run(ANALYZER, place, "--train", "small_1.nn", "train.txt")
    assert (folder / "small_1.nn").read_bytes() == (other / "small_1.nn").read_bytes()


def test_programs_work_from_another_folder(folder, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    assert run(GENERATOR, elsewhere, str(folder / "small.conf"), "1").returncode == 0
    predicted = run(ANALYZER, elsewhere, "--predict", str(elsewhere / "small_1.nn"), str(folder / "boards.txt"))
    assert predicted.returncode == 0
    assert len(predicted.stdout.splitlines()) == len(BOARDS)
