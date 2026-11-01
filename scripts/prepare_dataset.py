"""
Dataset preparation

Merges every chessboard file of the dataset folder, removes the duplicates
and shuffles the chessboards, so the network does not see them sorted by
answer. The same seed always gives the same file.

Run it from the root of the repository:
    python3 -m scripts.prepare_dataset [--dataset FOLDER] [--output FILE] [--seed N]
"""

import argparse
import glob
import os
import random
import sys
from collections import Counter

from analyzer.chessfile import read_chessfile

EXIT_OK = 0
EXIT_ERROR = 84

DATASET_FOLDER = "dataset"
OUTPUT_FILE = os.path.join(DATASET_FOLDER, "clean.txt")
SEED = 42


def find_files(folder: str) -> list[str]:
    """Find the chessboard files of the dataset, e.g. "dataset/check/10_pieces.txt".

    Only the files inside a sub-folder are taken, so the files written by
    this script at the top of the folder are never read again.

    Returns:
        The paths, sorted, so they are always read in the same order.

    Raises:
        ValueError: If there is no file.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*", "*.txt")))
    if not paths:
        raise ValueError(f"{folder}: no chessboard file found")
    return paths


def read_dataset(paths: list[str]) -> list[tuple[str, str]]:
    """Read every chessboard of the files, with its expected answer.

    Returns:
        (fen, answer) pairs, in the order of the files.

    Raises:
        ValueError: If a line is not a valid chessboard with a known answer.
    """
    boards = []
    for path in paths:
        fens, answers = read_chessfile(path, answers_needed=True)
        boards.extend(zip(fens, answers))
    return boards


def position_key(fen: str) -> str:
    """The part of a FEN that the network sees: the pieces and who plays.

    Two FENs that only differ by castling, en passant or move counters give
    the same inputs to the network, so they are the same chessboard for it.
    """
    pieces, side = fen.split()[:2]
    return f"{pieces} {side}"


def remove_duplicates(boards: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only the first chessboard of each position.

    Returns:
        The boards without duplicates, in the same order.
    """
    seen = set()
    unique = []
    for fen, answer in boards:
        key = position_key(fen)
        if key not in seen:
            seen.add(key)
            unique.append((fen, answer))
    return unique


def shuffle_boards(boards: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    """Return the boards in a random order, always the same for one seed."""
    shuffled = list(boards)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def write_boards(boards: list[tuple[str, str]], path: str) -> None:
    """Write one "FEN answer" line per chessboard, the format of --train."""
    with open(path, "w", encoding="utf-8") as file:
        for fen, answer in boards:
            file.write(f"{fen} {answer}\n")


def parse_args(args: list[str]) -> argparse.Namespace:
    """Read the options of the script."""
    parser = argparse.ArgumentParser(prog="python3 -m scripts.prepare_dataset",
                                     description="Merge, clean and shuffle the chessboard dataset.")
    parser.add_argument("--dataset", default=DATASET_FOLDER, help="folder with the chessboard files")
    parser.add_argument("--output", default=OUTPUT_FILE, help="file to write")
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the shuffle")
    return parser.parse_args(args)


def main(args: list[str]) -> int:
    """Prepare the dataset and print what was done. Returns the exit code."""
    options = parse_args(args)
    try:
        paths = find_files(options.dataset)
        boards = read_dataset(paths)
        unique = remove_duplicates(boards)
        write_boards(shuffle_boards(unique, options.seed), options.output)
    except (ValueError, OSError) as error:
        print(f"prepare_dataset: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Read {len(boards)} chessboards from {len(paths)} files")
    print(f"Removed {len(boards) - len(unique)} duplicates")
    print(f"Wrote {len(unique)} chessboards to {options.output} (seed {options.seed})")
    for answer, count in Counter(answer for _, answer in unique).most_common():
        print(f"    {answer:<16} {count:>7}  {count / len(unique):6.1%}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
