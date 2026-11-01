"""
Chessboard encoding tests
"""

import numpy as np
import pytest

from analyzer.encoding import INPUT_SIZE, encode_board, encode_boards

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHECK = "8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White"


def position(piece_plane, rank_index, file_index):
    return piece_plane * 64 + rank_index * 8 + file_index


def test_encoding_has_769_numbers_of_one_byte():
    inputs = encode_board(START)
    assert INPUT_SIZE == 769
    assert inputs.shape == (769,)
    assert inputs.dtype == np.uint8


def test_encoding_only_holds_zeros_and_ones():
    assert set(np.unique(encode_board(START))) <= {0, 1}


def test_encoding_of_the_start_has_one_number_per_piece():
    assert encode_board(START)[:768].sum() == 32


def test_encoding_puts_each_piece_in_its_grid():
    inputs = encode_board(START)
    assert inputs[position(0, 6, 4)] == 1
    assert inputs[position(5, 7, 4)] == 1
    assert inputs[position(11, 0, 4)] == 1
    assert inputs[position(6, 1, 0)] == 1
    assert inputs[position(0, 4, 4)] == 0


def test_encoding_of_a_dataset_line():
    inputs = encode_board(CHECK)
    assert inputs[:768].sum() == 6
    assert inputs[position(3, 2, 0)] == 1
    assert inputs[position(11, 2, 3)] == 1
    assert inputs[position(5, 5, 5)] == 1


def test_last_number_is_the_side_to_move():
    assert encode_board(START)[-1] == 1
    assert encode_board(CHECK)[-1] == 0


def test_only_the_side_to_move_changes_the_last_number():
    white = encode_board("8/8/8/8/8/8/8/k1K5 w - - 0 1")
    black = encode_board("8/8/8/8/8/8/8/k1K5 b - - 0 1")
    np.testing.assert_array_equal(white[:768], black[:768])
    assert (white[-1], black[-1]) == (1, 0)


def test_encode_boards_gives_one_row_per_board():
    inputs = encode_boards([START, CHECK, START])
    assert inputs.shape == (3, 769)
    assert inputs.dtype == np.uint8
    np.testing.assert_array_equal(inputs[1], encode_board(CHECK))


def test_encode_boards_of_nothing_gives_an_empty_array():
    assert encode_boards([]).shape == (0, 769)


def test_encoding_an_invalid_fen_raises_value_error():
    with pytest.raises(ValueError, match="invalid FEN"):
        encode_boards([START, "not a chessboard"])
