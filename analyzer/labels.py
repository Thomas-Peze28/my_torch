"""
Answer labels
"""

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]


def label_to_index(label: str) -> int:
    """Give the number of an answer, used to train the network.

    The order of LABELS must never change once a network is trained:
    output 1 of the network always means "Check White".

    Args:
        label: One of LABELS, e.g. "Check White".

    Returns:
        Its index in LABELS, from 0 to 4.

    Raises:
        ValueError: If the answer is unknown.
    """
    if label not in LABELS:
        raise ValueError(f'unknown answer "{label}"')
    return LABELS.index(label)


def index_to_label(index: int) -> str:
    """Give the answer text of a network output, used to print predictions.

    Args:
        index: An index from 0 to 4, e.g. given by Network.predict().

    Returns:
        The answer, e.g. "Check White".

    Raises:
        ValueError: If the index is not between 0 and 4.
    """
    if not 0 <= index < len(LABELS):
        raise ValueError(f"unknown answer number {index}")
    return LABELS[index]
