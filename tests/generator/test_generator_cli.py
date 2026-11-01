"""
my_torch_generator tests
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from generator.cli import UsageError, output_name, parse_args
from my_torch import Network

ROOT = Path(__file__).resolve().parent.parent.parent
PROGRAM = ROOT / "my_torch_generator"
CONFIGS = ROOT / "configs"


def run(folder, *args):
    """Run my_torch_generator in a folder, like the user would."""
    return subprocess.run([sys.executable, str(PROGRAM), *args],
                          capture_output=True, text=True, cwd=folder)


def networks_in(folder):
    return sorted(path.name for path in Path(folder).glob("*.nn"))


@pytest.fixture
def folder(tmp_path):
    """A folder holding copies of the example configuration files."""
    for config in CONFIGS.glob("*.conf"):
        shutil.copy(config, tmp_path)
    return tmp_path


def write_config(folder, name, description):
    (folder / name).write_text(json.dumps(description))


def test_help_prints_usage_and_returns_0(folder):
    result = run(folder, "--help")
    assert result.returncode == 0
    assert "USAGE" in result.stdout
    assert networks_in(folder) == []


def test_generator_creates_the_asked_networks(folder):
    result = run(folder, "basic_network.conf", "3", "chess_network.conf", "1")
    assert result.returncode == 0
    assert result.stderr == ""
    assert networks_in(folder) == ["basic_network_1.nn", "basic_network_2.nn",
                                   "basic_network_3.nn", "chess_network_1.nn"]


def test_generated_networks_load_with_their_layers_and_settings(folder):
    run(folder, "chess_network.conf", "1")
    description = json.loads((folder / "chess_network.conf").read_text())

    network = Network.load(str(folder / "chess_network_1.nn"))

    assert [layer.config() for layer in network.layers] == description["layers"]
    assert network.settings == description["settings"]
    probabilities = network.forward(np.zeros((3, 769)))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))


def test_generated_networks_have_different_weights(folder):
    run(folder, "basic_network.conf", "2")
    first = Network.load(str(folder / "basic_network_1.nn"))
    second = Network.load(str(folder / "basic_network_2.nn"))
    assert not np.array_equal(first.layers[0].weights, second.layers[0].weights)


def test_generation_with_a_seed_is_repeatable(folder, tmp_path_factory):
    other = tmp_path_factory.mktemp("again")
    shutil.copy(folder / "basic_network.conf", other)
    run(folder, "basic_network.conf", "1")
    run(other, "basic_network.conf", "1")
    first = Network.load(str(folder / "basic_network_1.nn"))
    second = Network.load(str(other / "basic_network_1.nn"))
    for (first_value, _), (second_value, _) in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(first_value, second_value)


def test_generation_without_a_seed_works(folder):
    write_config(folder, "no_seed.conf", {"layers": [{"type": "Linear", "input_size": 2, "output_size": 1}]})
    assert run(folder, "no_seed.conf", "2").returncode == 0
    assert networks_in(folder) == ["no_seed_1.nn", "no_seed_2.nn"]


@pytest.mark.parametrize("args", [
    [],
    ["basic_network.conf"],
    ["basic_network.conf", "0"],
    ["basic_network.conf", "-1"],
    ["basic_network.conf", "abc"],
    ["basic_network.conf", "2.5"],
    ["missing.conf", "2"],
    ["broken.conf", "1"],
    ["sizes.conf", "1"],
    ["unknown.conf", "1"],
    ["seed.conf", "1"],
    ["basic_network.conf", "2", "sizes.conf", "1"],
])
def test_invalid_arguments_return_84_and_write_nothing(folder, args):
    (folder / "broken.conf").write_text("{oops")
    write_config(folder, "sizes.conf", {"layers": [
        {"type": "Linear", "input_size": 769, "output_size": 64},
        {"type": "Linear", "input_size": 32, "output_size": 5}]})
    write_config(folder, "unknown.conf", {"layers": [{"type": "Conv"}]})
    write_config(folder, "seed.conf", {"layers": [{"type": "Linear", "input_size": 2, "output_size": 1}],
                                       "settings": {"seed": "abc"}})

    result = run(folder, *args)

    assert result.returncode == 84
    assert result.stderr != ""
    assert result.stdout == ""
    assert networks_in(folder) == []


def test_parse_args_gives_file_and_count_pairs():
    assert parse_args(["a.conf", "3", "b.conf", "1"]) == [("a.conf", 3), ("b.conf", 1)]


def test_parse_args_rejects_odd_arguments():
    with pytest.raises(UsageError):
        parse_args(["a.conf", "3", "b.conf"])


def test_output_name_drops_the_folder_and_the_extension():
    assert output_name("configs/basic_network.conf", 2) == "basic_network_2.nn"
    assert output_name("chess.json", 10) == "chess_10.nn"
