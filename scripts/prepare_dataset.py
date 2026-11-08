"""
Dataset preparation

Merges every chessboard file of the dataset folder, removes the duplicates
and shuffles the chessboards, so the network does not see them sorted by
answer. Then it splits them into three files:
    train.txt       80% to learn
    validation.txt  10% to check the network during training
    test.txt        10% for the final mark, never used to choose anything
With --mirror, train.txt also gets every training chessboard seen in a
mirror (left and right swapped), which has the same answer: twice more
examples to learn from. validation.txt and test.txt do not change.
The same seed always gives the same files.

Run it from the root of the repository:
    python3 -m scripts.prepare_dataset [--dataset FOLDER] [--output FOLDER] [--seed N] [--mirror]
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
SEED = 42
VALIDATION_SHARE = 0.1
TEST_SHARE = 0.1

Board = tuple[str, str]
"""A chessboard and its expected answer: (FEN, answer)."""


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


def read_dataset(paths: list[str]) -> list[Board]:
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


def remove_duplicates(boards: list[Board]) -> list[Board]:
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


def mirror_fen(fen: str) -> str:
    """The same chessboard seen in a mirror: column a becomes h, b becomes g...

    Each rank is read backwards; "3" (three empty squares) stays one
    character, so this works on the FEN text. A check or a checkmate stays
    the same, by the same colour. Castling is removed, since the kings and
    rooks are no longer on their starting squares; the en passant square is
    mirrored.

    Example:
        "r3k2r/8/8/8/4P3/8/8/R3K2R b KQkq e3 0 1"
        gives "r2k3r/8/8/8/3P4/8/8/R2K3R b - d3 0 1".
    """
    pieces, side, _, en_passant, *counters = fen.split()
    pieces = "/".join(rank[::-1] for rank in pieces.split("/"))
    if en_passant != "-":
        en_passant = "abcdefgh"["hgfedcba".index(en_passant[0])] + en_passant[1:]
    return " ".join([pieces, side, "-", en_passant, *counters])


def add_mirrors(train: list[Board], others: list[Board]) -> list[Board]:
    """Add the mirror of every training chessboard to the training part.

    A mirror is not added if the network would already see it: a position
    that is already in the training part (a symmetric chessboard is its own
    mirror), or one that is in the validation or test part, which would let
    the network learn a chessboard it is marked on.

    Args:
        train: The training boards.
        others: The validation and test boards.

    Returns:
        The training boards followed by their new mirrors.
    """
    seen = {position_key(fen) for fen, _ in train + others}
    mirrors = []
    for fen, answer in train:
        mirror = mirror_fen(fen)
        if position_key(mirror) not in seen:
            seen.add(position_key(mirror))
            mirrors.append((mirror, answer))
    return train + mirrors


def shuffle_boards(boards: list[Board], seed: int) -> list[Board]:
    """Return the boards in a random order, always the same for one seed."""
    shuffled = list(boards)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def split_boards(boards: list[Board], validation_share: float = VALIDATION_SHARE,
                 test_share: float = TEST_SHARE) -> dict[str, list[Board]]:
    """Cut the boards into a training, a validation and a test part.

    The boards must already be shuffled and without duplicates: the parts
    are consecutive slices, so no chessboard can be in two parts.

    Args:
        boards: The shuffled boards.
        validation_share: Part of the boards used for validation (0.1 = 10%).
        test_share: Part of the boards kept for the final test (0.1 = 10%).

    Returns:
        {"train": [...], "validation": [...], "test": [...]}.
    """
    validation_size = round(len(boards) * validation_share)
    test_size = round(len(boards) * test_share)
    train_size = len(boards) - validation_size - test_size
    return {
        "train": boards[:train_size],
        "validation": boards[train_size:train_size + validation_size],
        "test": boards[train_size + validation_size:],
    }


def write_boards(boards: list[Board], path: str) -> None:
    """Write one "FEN answer" line per chessboard, the format of --train."""
    with open(path, "w", encoding="utf-8") as file:
        for fen, answer in boards:
            file.write(f"{fen} {answer}\n")


def print_summary(boards: list[Board], parts: dict[str, list[Board]],
                  folder: str) -> None:
    """Print the size of each file and the share of each answer in it."""
    columns = {"all": boards} | parts
    counts = {name: Counter(answer for _, answer in column) for name, column in columns.items()}
    print(f"{'':<16}" + "".join(f" {name:>11}" for name in columns))
    print(f"{'chessboards':<16}" + "".join(f" {len(column):>11}" for column in columns.values()))
    for answer, _ in counts["all"].most_common():
        shares = [counts[name][answer] / max(len(column), 1) for name, column in columns.items()]
        print(f"{answer:<16}" + "".join(f" {share:>11.1%}" for share in shares))
    print("Files: " + ", ".join(os.path.join(folder, f"{name}.txt") for name in parts))


def parse_args(args: list[str]) -> argparse.Namespace:
    """Read the options of the script."""
    parser = argparse.ArgumentParser(prog="python3 -m scripts.prepare_dataset",
                                     description="Merge, clean, shuffle and split the chessboard dataset.")
    parser.add_argument("--dataset", default=DATASET_FOLDER, help="folder with the chessboard files")
    parser.add_argument("--output", default=DATASET_FOLDER, help="folder where the three files are written")
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the shuffle")
    parser.add_argument("--mirror", action="store_true",
                        help="add the mirror of every training chessboard to train.txt")
    return parser.parse_args(args)


def main(args: list[str]) -> int:
    """Prepare the dataset and print what was done. Returns the exit code."""
    options = parse_args(args)
    try:
        paths = find_files(options.dataset)
        boards = read_dataset(paths)
        unique = remove_duplicates(boards)
        parts = split_boards(shuffle_boards(unique, options.seed))
        if options.mirror:
            original = len(parts["train"])
            mirrored = add_mirrors(parts["train"], parts["validation"] + parts["test"])
            parts["train"] = shuffle_boards(mirrored, options.seed)
        for name, part in parts.items():
            write_boards(part, os.path.join(options.output, f"{name}.txt"))
    except (ValueError, OSError) as error:
        print(f"prepare_dataset: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Read {len(boards)} chessboards from {len(paths)} files")
    print(f"Removed {len(boards) - len(unique)} duplicates")
    print(f"Shuffled {len(unique)} chessboards with seed {options.seed}")
    if options.mirror:
        print(f"Added {len(parts['train']) - original} mirrored chessboards to train.txt")
    print_summary(unique, parts, options.output)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
