"""
Network evaluation

Shows which answers a trained network finds and which ones it mixes up,
on a chessboard file it has never learned from:
- the accuracy for each answer, printed as a table;
- the confusion matrix, drawn as a PNG image: each row is the expected
  answer, each column the answer of the network, so the diagonal holds the
  right answers and every other cell a mix-up.

Run it from the root of the repository:
    python3 -m scripts.evaluate_network NETWORK [--file FILE] [--output FILE.png]
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzer.chess_network import check_chess_network
from analyzer.chessfile import read_chessfile
from analyzer.labels import LABELS, label_to_index
from analyzer.predict import predict_answers
from my_torch import Network

EXIT_OK = 0
EXIT_ERROR = 84

CHESS_FILE = "dataset/validation.txt"


def confusion_matrix(expected: list[str], answered: list[str]) -> np.ndarray:
    """Count how often each expected answer got each answer of the network.

    Returns:
        Array of shape (5, 5): cell [i, j] is the number of chessboards
        whose expected answer is LABELS[i] and that the network answered
        LABELS[j].
    """
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for wanted, given in zip(expected, answered):
        matrix[label_to_index(wanted), label_to_index(given)] += 1
    return matrix


def answer_accuracy(matrix: np.ndarray) -> np.ndarray:
    """Share of the chessboards of each expected answer that are found.

    Returns:
        One number between 0 and 1 per answer, in the order of LABELS
        (0 for an answer that never appears).
    """
    totals = matrix.sum(axis=1)
    return np.divide(matrix.diagonal(), totals, out=np.zeros(len(LABELS)), where=totals > 0)


def plot_confusion(matrix: np.ndarray, output: str, title: str) -> None:
    """Draw the confusion matrix on a PNG image.

    The color of a cell is the share of its row, so rare answers are as
    readable as frequent ones; the number written is the count.
    """
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(shares * 100, cmap="Greens", vmin=0, vmax=100)
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            color = "white" if shares[row, column] > 0.6 else "black"
            axis.text(column, row, f"{matrix[row, column]}\n{shares[row, column]:.1%}",
                      ha="center", va="center", color=color, fontsize=8)
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("answer of the network")
    axis.set_ylabel("expected answer")
    axis.set_title(title)
    figure.colorbar(image, label="share of the expected answer (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


def parse_args(args: list[str]) -> argparse.Namespace:
    """Read the options of the script."""
    parser = argparse.ArgumentParser(prog="python3 -m scripts.evaluate_network",
                                     description="Measure a chess network for each answer.")
    parser.add_argument("network", help="trained network file")
    parser.add_argument("--file", default=CHESS_FILE, help="chessboards with their expected answers")
    parser.add_argument("--output", help="PNG file where to draw the confusion matrix")
    return parser.parse_args(args)


def main(args: list[str]) -> int:
    """Measure the network, print the table and draw the matrix. Returns the exit code."""
    options = parse_args(args)
    try:
        network = Network.load(options.network)
        check_chess_network(network)
        fens, expected = read_chessfile(options.file, answers_needed=True)
        if not fens:
            raise ValueError(f"{options.file}: no chessboard to check")
        matrix = confusion_matrix(expected, predict_answers(network, fens))
        if options.output:
            plot_confusion(matrix, options.output, f"{options.network} on {options.file}")
    except (ValueError, OSError) as error:
        print(f"evaluate_network: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(f"{'answer':<16} {'chessboards':>11} {'found':>8}")
    for label, total, share in zip(LABELS, matrix.sum(axis=1), answer_accuracy(matrix)):
        print(f"{label:<16} {total:>11} {share:>8.1%}")
    print(f"{'all':<16} {matrix.sum():>11} {matrix.trace() / matrix.sum():>8.1%}")
    if options.output:
        print(f"Confusion matrix written to {options.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
