"""
Learning curves

Draws the loss and the accuracy of one or several trainings, epoch by
epoch, from the history files written by "train_network --history". With
one file, it shows how a network learns; with several, it compares them.
Dashed lines are the chessboards the network learns from, solid lines the
chessboards put aside to check it, and a dot marks the epoch that is kept.

Run it from the root of the repository:
    python3 -m scripts.plot_history HISTORY [HISTORY ...] --output FILE.png [--title TEXT]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXIT_OK = 0
EXIT_ERROR = 84


def read_history(path: str) -> dict:
    """Read the history of a training from a file written by train_network.

    Raises:
        ValueError: If the file is not a history file.
    """
    try:
        with open(path, encoding="utf-8") as file:
            content = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError(f"{path}: not a training history file") from None
    history = content.get("history") if isinstance(content, dict) else None
    if not isinstance(history, dict) or "loss" not in history or "accuracy" not in history:
        raise ValueError(f"{path}: not a training history file")
    return history


def name_of(path: str) -> str:
    """Name shown in the legend: "benchmarks/init_he.json" gives "init_he"."""
    return os.path.splitext(os.path.basename(path))[0]


def plot_histories(paths: list[str], output: str, title: str = "") -> None:
    """Draw the loss and the accuracy of every history on one PNG image.

    Args:
        paths: History files written by train_network --history.
        output: The PNG file to write.
        title: Title written above the two graphs.

    Raises:
        ValueError: If a file is not a history file.
    """
    histories = [read_history(path) for path in paths]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for index, (path, history) in enumerate(zip(paths, histories)):
        color = f"C{index}"
        epochs = range(1, len(history["loss"]) + 1)
        loss_axis.plot(epochs, history["loss"], "--", color=color)
        accuracy_axis.plot(epochs, [value * 100 for value in history["accuracy"]], "--", color=color)
        if history.get("val_loss"):
            loss_axis.plot(epochs, history["val_loss"], color=color)
            accuracy_axis.plot(epochs, [value * 100 for value in history["val_accuracy"]], color=color,
                               label=name_of(path))
            best = history["best_epoch"]
            loss_axis.plot(best, history["val_loss"][best - 1], "o", color=color, markersize=7)
            accuracy_axis.plot(best, history["val_accuracy"][best - 1] * 100, "o", color=color, markersize=7)

    loss_axis.set_title("Loss (lower is better)")
    loss_axis.set_ylabel("cross-entropy loss")
    accuracy_axis.set_title("Accuracy (higher is better)")
    accuracy_axis.set_ylabel("right answers (%)")
    for axis in (loss_axis, accuracy_axis):
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    legend = [plt.Line2D([], [], color="gray", linestyle="--", label="train (learned chessboards)"),
              plt.Line2D([], [], color="gray", label="validation (chessboards put aside)"),
              plt.Line2D([], [], color="gray", marker="o", linestyle="", label="kept epoch")]
    if len(paths) > 1:
        legend += [plt.Line2D([], [], color=f"C{index}", linewidth=4, label=name_of(path))
                   for index, path in enumerate(paths)]
    figure.legend(handles=legend, loc="lower center", ncol=min(len(legend), 4), fontsize="small", frameon=False)
    if title:
        figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08 if len(paths) > 1 else 0.05, 1, 1))
    figure.savefig(output, dpi=120)
    plt.close(figure)


def parse_args(args: list[str]) -> argparse.Namespace:
    """Read the options of the script."""
    parser = argparse.ArgumentParser(prog="python3 -m scripts.plot_history",
                                     description="Draw the learning curves of one or several trainings.")
    parser.add_argument("histories", nargs="+", help="history files written by train_network --history")
    parser.add_argument("--output", required=True, help="PNG file to write")
    parser.add_argument("--title", default="", help="title of the image")
    return parser.parse_args(args)


def main(args: list[str]) -> int:
    """Draw the curves and return the exit code."""
    options = parse_args(args)
    try:
        plot_histories(options.histories, options.output, options.title)
    except (ValueError, OSError) as error:
        print(f"plot_history: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Curves written to {options.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
