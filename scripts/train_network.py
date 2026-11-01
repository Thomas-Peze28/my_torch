"""
Network training

Does in one command what a user would do with the two programs:
    ./my_torch_generator CONFIG 1
    ./my_torch_analyzer --train NETWORK dataset/train.txt
then gives the accuracy of the trained network on the validation file. The
seed of the configuration fixes the starting weights and the order of the
mini-batches, so the same command always gives the same network.

Run it from the root of the repository, after "make dataset":
    python3 -m scripts.train_network [--config FILE] [--dataset FOLDER] [--output FILE] [--test]
"""

import argparse
import os
import sys
import time

from analyzer.chessfile import read_chessfile
from analyzer.predict import predict_answers
from analyzer.train import run_train
from generator.cli import UsageError, build_networks
from my_torch import Network

EXIT_OK = 0
EXIT_ERROR = 84

CONFIG_FILE = os.path.join("configs", "chess_network.conf")
DATASET_FOLDER = "dataset"
OUTPUT_FILE = "my_torch_network.nn"


def accuracy(network: Network, path: str) -> float:
    """Share of the chessboards of a file that the network answers right.

    Args:
        network: A chess network.
        path: A chessboard file where every line has its expected answer.

    Returns:
        A number between 0 and 1 (0.9 = 90% of right answers).

    Raises:
        ValueError: If the file has no chessboard or a broken line.
    """
    fens, expected = read_chessfile(path, answers_needed=True)
    if not fens:
        raise ValueError(f"{path}: no chessboard to check")
    answers = predict_answers(network, fens)
    right = sum(1 for answer, wanted in zip(answers, expected) if answer == wanted)
    return right / len(fens)


def parse_args(args: list[str]) -> argparse.Namespace:
    """Read the options of the script."""
    parser = argparse.ArgumentParser(prog="python3 -m scripts.train_network",
                                     description="Create a chess network, train it and measure it.")
    parser.add_argument("--config", default=CONFIG_FILE, help="configuration file of the network")
    parser.add_argument("--dataset", default=DATASET_FOLDER,
                        help="folder with train.txt, validation.txt and test.txt")
    parser.add_argument("--output", default=OUTPUT_FILE, help="where to save the trained network")
    parser.add_argument("--test", action="store_true",
                        help="also give the final mark on test.txt (only for the chosen network)")
    return parser.parse_args(args)


def main(args: list[str]) -> int:
    """Create, train and measure a network. Returns the exit code."""
    options = parse_args(args)
    try:
        _, network = build_networks(options.config, 1)[0]
        network.save(options.output)
        print(f"Created {options.output} from {options.config}")

        start = time.time()
        run_train(options.output, os.path.join(options.dataset, "train.txt"))
        print(f"Training took {(time.time() - start) / 60:.1f} minutes")

        trained = Network.load(options.output)
        validation = accuracy(trained, os.path.join(options.dataset, "validation.txt"))
        print(f"Validation accuracy: {validation:.2%}")
        if options.test:
            print(f"Test accuracy: {accuracy(trained, os.path.join(options.dataset, 'test.txt')):.2%}")
    except (UsageError, ValueError, OSError) as error:
        print(f"train_network: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
