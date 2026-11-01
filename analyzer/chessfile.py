"""
Chessboard file reading
"""

from analyzer.fen import read_fen
from analyzer.labels import LABELS

FEN_FIELDS = 6


def read_chessfile(path: str, answers_needed: bool) -> tuple[list[str], list[str]]:
    """Read every chessboard of a file, and its expected answer if any.

    Each line is a FEN (its 6 fields), optionally followed by the expected
    answer, e.g. "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White".
    Empty lines are skipped. Every line is checked before returning, so an
    error is found before anything is printed or trained.

    Args:
        path: The file to read.
        answers_needed: True in train mode: every line must end with a known
            answer. False in predict mode: answers are allowed and not checked.

    Returns:
        (fens, answers): one FEN per chessboard, and its answer ("" if none),
        in the order of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not text, or a line has an invalid FEN,
            or a missing or unknown answer when answers are needed. The
            message gives the line number.
    """
    fens = []
    answers = []
    try:
        with open(path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                fields = line.split()
                if not fields:
                    continue
                fen = " ".join(fields[:FEN_FIELDS])
                answer = " ".join(fields[FEN_FIELDS:])
                try:
                    read_fen(fen)
                except ValueError as error:
                    raise ValueError(f"{path} line {line_number}: {error}") from None
                if answers_needed and not answer:
                    raise ValueError(f"{path} line {line_number}: missing expected answer")
                if answers_needed and answer not in LABELS:
                    raise ValueError(f'{path} line {line_number}: unknown answer "{answer}"')
                fens.append(fen)
                answers.append(answer)
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not a text file") from None
    return fens, answers
