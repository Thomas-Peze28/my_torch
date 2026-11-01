"""
Command line of my_torch_generator
"""

import json
import os
import sys

from my_torch import Network

EXIT_OK = 0
EXIT_ERROR = 84

HELP = """USAGE

    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION

    config_file_i    Configuration file containing description of a neural network we want
                     to generate.

    nb_i             Number of neural networks to generate based on the configuration file."""


class UsageError(Exception):
    """Raised when the arguments or a configuration file are not valid."""


def parse_args(args: list[str]) -> list[tuple[str, int]]:
    """Turn the arguments into (configuration file, number of networks) pairs.

    Args:
        args: The program arguments, without the program name.

    Returns:
        One (path, count) pair per configuration file, in order.

    Raises:
        UsageError: If the arguments are not pairs of a file and a positive
            whole number.
    """
    if not args or len(args) % 2 != 0:
        raise UsageError("expected pairs of config_file and nb")
    jobs = []
    for index in range(0, len(args), 2):
        path, count = args[index], args[index + 1]
        if not count.isdigit() or int(count) <= 0:
            raise UsageError(f'"{count}" is not a positive whole number')
        jobs.append((path, int(count)))
    return jobs


def read_config(path: str) -> dict:
    """Read a JSON configuration file.

    Raises:
        UsageError: If the file is missing or is not valid JSON.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise UsageError(f"{path}: no such file") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UsageError(f"{path}: not a valid JSON file ({error})") from None


def output_name(path: str, number: int) -> str:
    """Name of the n-th network made from a configuration file.

    "configs/basic_network.conf" and 2 give "basic_network_2.nn".
    """
    base = os.path.splitext(os.path.basename(path))[0]
    return f"{base}_{number}.nn"


def build_networks(path: str, count: int) -> list[tuple[str, Network]]:
    """Build the networks of one configuration file, without saving them.

    If the settings hold a "seed", network n uses seed + n, so the networks
    differ from each other but are the same every time.

    Raises:
        UsageError: If the configuration is not valid.
    """
    description = read_config(path)
    settings = description.get("settings", {}) if isinstance(description, dict) else {}
    seed = settings.get("seed") if isinstance(settings, dict) else None
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise UsageError(f'{path}: "seed" must be a whole number')
    networks = []
    for number in range(1, count + 1):
        try:
            network = Network.from_config(description, seed=None if seed is None else seed + number)
        except ValueError as error:
            raise UsageError(f"{path}: {error}") from None
        networks.append((output_name(path, number), network))
    return networks


def main(args: list[str]) -> int:
    """Run the program and return its exit code (0 if OK, 84 on error).

    Every configuration file is checked before any network file is
    written, so an error never leaves half of the files behind.
    """
    if args in (["-h"], ["--help"]):
        print(HELP)
        return EXIT_OK
    try:
        networks = []
        for path, count in parse_args(args):
            networks.extend(build_networks(path, count))
        for name, network in networks:
            network.save(name)
    except UsageError as error:
        print(f"my_torch_generator: {error}", file=sys.stderr)
        print("Try './my_torch_generator --help' for more information.", file=sys.stderr)
        return EXIT_ERROR
    except OSError as error:
        print(f"my_torch_generator: cannot write the network: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
