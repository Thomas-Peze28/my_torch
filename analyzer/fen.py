"""
FEN reading
"""

PIECES = "PNBRQKpnbrqk"
EMPTY = "."


def read_fen(text: str) -> tuple[list[str], str]:
    """Read the pieces and the side to move of a chessboard written in FEN.

    A FEN has 6 fields separated by spaces, e.g.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".
    Only the first two are used: the pieces and the side to move. The
    other fields (castling, en passant, move counters) and anything after
    them, like an expected answer, are ignored.

    Args:
        text: A line starting with a FEN.

    Returns:
        (board, side): 8 ranks from rank 8 to rank 1, and "w" or "b".

    Raises:
        ValueError: If the pieces or the side to move are not valid.
    """
    fields = text.split()
    if len(fields) < 2:
        raise ValueError(f'invalid FEN "{text.strip()}": the pieces and the side to move are needed')
    placement, side = fields[0], fields[1]

    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f'invalid FEN "{placement}": 8 ranks are needed, not {len(ranks)}')
    board = [_read_rank(rank, placement) for rank in ranks]

    if side not in ("w", "b"):
        raise ValueError(f'invalid FEN side to move "{side}": "w" or "b" is needed')
    return board, side


def _read_rank(rank: str, placement: str) -> str:
    """Turn one rank of a FEN, like "4P3", into 8 squares, like "....P...".

    Raises:
        ValueError: If a character is unknown or the rank is not 8 squares long.
    """
    squares = ""
    for character in rank:
        if character in "12345678":
            squares += EMPTY * int(character)
        elif character in PIECES:
            squares += character
        else:
            raise ValueError(f'invalid FEN "{placement}": unknown character "{character}"')
    if len(squares) != 8:
        raise ValueError(f'invalid FEN "{placement}": rank "{rank}" is not 8 squares long')
    return squares
