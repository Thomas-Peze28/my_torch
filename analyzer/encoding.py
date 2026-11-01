"""
Chessboard encoding
"""

import numpy as np

from analyzer.fen import EMPTY, PIECES, read_fen

SQUARES = 64
INPUT_SIZE = len(PIECES) * SQUARES + 1


def encode_board(text: str) -> np.ndarray:
    """Encode a FEN position as 768 piece values and the side to move.

    Args:
        text: A FEN string.

    Returns:
        A uint8 array of shape ``(769,)``.

    Raises:
        ValueError: If the FEN is invalid.
    """
    board, side = read_fen(text)
    inputs = np.zeros(INPUT_SIZE, dtype=np.uint8)
    for rank_index, rank in enumerate(board):
        for file_index, square in enumerate(rank):
            if square != EMPTY:
                plane = PIECES.index(square)
                inputs[plane * SQUARES + rank_index * 8 + file_index] = 1
    inputs[-1] = 1 if side == "w" else 0
    return inputs


def encode_boards(texts: list[str]) -> np.ndarray:
    """Turn many chessboards written in FEN into one array, one row per board.

    Args:
        texts: Lines starting with a FEN.

    Returns:
        Array of shape (number of boards, 769) and type uint8.

    Raises:
        ValueError: If one of the FENs is not valid.
    """
    inputs = np.zeros((len(texts), INPUT_SIZE), dtype=np.uint8)
    for row, text in enumerate(texts):
        inputs[row] = encode_board(text)
    return inputs
