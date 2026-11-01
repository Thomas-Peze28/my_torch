"""
FEN reading tests
"""

import pytest

from analyzer.fen import read_fen

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_read_fen_gives_the_starting_position():
    board, side = read_fen(START)
    assert board == [
        "rnbqkbnr",
        "pppppppp",
        "........",
        "........",
        "........",
        "........",
        "PPPPPPPP",
        "RNBQKBNR",
    ]
    assert side == "w"


def test_read_fen_places_the_kings_on_e1_and_e8():
    board, _ = read_fen(START)
    assert board[7][4] == "K"
    assert board[0][4] == "k"


def test_read_fen_expands_the_empty_squares():
    board, side = read_fen("8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59")
    assert board[2] == "R..k...."
    assert board[3] == "....r.p."
    assert side == "b"


def test_read_fen_ignores_the_expected_answer_after_the_fen():
    board, side = read_fen("8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White")
    assert board[2] == "R..k...."
    assert side == "b"


def test_read_fen_only_needs_the_pieces_and_the_side():
    board, side = read_fen("8/8/8/8/8/8/8/k1K5 b")
    assert board[7] == "k.K....."
    assert side == "b"


@pytest.mark.parametrize("text", [
    "",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQXBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR z KQkq - 0 1",
])
def test_read_fen_rejects_an_invalid_fen(text):
    with pytest.raises(ValueError, match="invalid FEN"):
        read_fen(text)
