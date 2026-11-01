"""
Chess network check
"""

from analyzer.encoding import INPUT_SIZE
from analyzer.labels import LABELS
from my_torch import Linear, Network


def check_chess_network(network: Network) -> None:
    """Make sure a network reads a chessboard and gives one of the answers.

    Its first Linear layer must take INPUT_SIZE (769) inputs, and its last
    Linear layer must give one output per answer (5).

    Raises:
        ValueError: If the network does not have these sizes.
    """
    linears = [layer for layer in network.layers if isinstance(layer, Linear)]
    if (not linears
            or linears[0].config()["input_size"] != INPUT_SIZE
            or linears[-1].config()["output_size"] != len(LABELS)):
        raise ValueError(f"this network does not take a chessboard ({INPUT_SIZE} inputs) "
                         f"and give {len(LABELS)} answers")
