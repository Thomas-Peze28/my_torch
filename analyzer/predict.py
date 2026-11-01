"""
Predict mode
"""

from analyzer.chess_network import check_chess_network
from analyzer.chessfile import read_chessfile
from analyzer.encoding import encode_boards
from analyzer.labels import index_to_label
from my_torch import Network

CHUNK_SIZE = 10000


def predict_answers(network: Network, fens: list[str]) -> list[str]:
    """Give the answer of the network for every chessboard, in order.

    The chessboards are encoded and predicted CHUNK_SIZE at a time, so a
    big file does not need gigabytes of memory.

    Args:
        network: A chess network.
        fens: The chessboards, written in FEN.

    Returns:
        One answer per chessboard, e.g. "Check White".
    """
    answers = []
    for start in range(0, len(fens), CHUNK_SIZE):
        inputs = encode_boards(fens[start:start + CHUNK_SIZE])
        answers.extend(index_to_label(index) for index in network.predict(inputs))
    return answers


def run_predict(loadfile: str, chessfile: str) -> None:
    """Print the answer of the network for every chessboard of a file.

    Everything is read and checked before the first answer is printed.

    Args:
        loadfile: The network file.
        chessfile: The chessboard file; expected answers are allowed and ignored.

    Raises:
        FileNotFoundError: If a file does not exist.
        ValueError: If the network or the chessboard file is not valid.
    """
    network = Network.load(loadfile)
    check_chess_network(network)
    fens, _ = read_chessfile(chessfile, answers_needed=False)
    answers = predict_answers(network, fens)
    if answers:
        print("\n".join(answers))
