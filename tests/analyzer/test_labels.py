"""
Answer labels tests
"""

import pytest

from analyzer.labels import LABELS, index_to_label, label_to_index


def test_labels_keep_their_order():
    assert LABELS == ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]


@pytest.mark.parametrize("index, label", list(enumerate(LABELS)))
def test_label_and_index_go_both_ways(index, label):
    assert label_to_index(label) == index
    assert index_to_label(index) == label


@pytest.mark.parametrize("label", ["Chek White", "check white", "Check", "Stalemate", ""])
def test_unknown_answer_raises_value_error(label):
    with pytest.raises(ValueError, match="unknown answer"):
        label_to_index(label)


@pytest.mark.parametrize("index", [-1, 5, 100])
def test_unknown_answer_number_raises_value_error(index):
    with pytest.raises(ValueError, match="unknown answer number"):
        index_to_label(index)


def test_index_to_label_accepts_numpy_numbers():
    import numpy as np

    assert index_to_label(np.int64(3)) == "Checkmate White"
