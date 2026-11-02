"""
Delivery tests

The grader looks for the pre-trained network by its name, which must start
with "my_torch_network". Any other file with such a name could be taken
instead and break the evaluation.
"""

import os
from pathlib import Path

from analyzer.chess_network import check_chess_network
from my_torch import Network

ROOT = Path(__file__).resolve().parent.parent
NOT_DELIVERED = {"dataset", "runs", "document", "__pycache__"}


def files_named_like_the_network():
    found = []
    for folder, subfolders, files in os.walk(ROOT):
        subfolders[:] = [name for name in subfolders if not name.startswith(".") and name not in NOT_DELIVERED]
        found += [Path(folder, name) for name in files if name.startswith("my_torch_network")]
    return found


def test_only_one_file_is_named_like_the_network():
    assert files_named_like_the_network() == [ROOT / "my_torch_network.nn"]


def test_the_pushed_network_is_a_chess_network():
    check_chess_network(Network.load(str(ROOT / "my_torch_network.nn")))
