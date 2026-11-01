import os
import sys

from analyzer.predict import run_predict
from analyzer.train import run_train

EXIT_OK = 0
EXIT_ERROR = 84

HELP = """USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION

    --train     Launch the neural network in training mode. Each chessboard in FILE must
                contain inputs to send to the neural network in FEN notation and the expected output
                separated by space. If specified, the newly trained neural network will be saved in
                SAVEFILE. Otherwise, it will be saved in the original LOADFILE.

    --predict   Launch the neural network in prediction mode. Each chessboard in FILE must
                contain inputs to send to the neural network in FEN notation, and optionally an expected
                output.

    --save      Save neural network into SAVEFILE. Only works in train mode.

    LOADFILE    File containing an artificial neural network

    CHESSFILE   File containing chessboards"""


class UsageError(Exception):
    """Raised when the program arguments are not valid."""


class Options:
    """What the user asked for, once the arguments are read."""

    def __init__(self, mode, loadfile, chessfile, savefile=None):
        self.mode = mode
        self.loadfile = loadfile
        self.chessfile = chessfile
        self.savefile = savefile


def parse_args(args):
    """Turn the list of arguments into Options.

    Raises UsageError if the arguments are not valid.
    """
    mode = None
    savefile = None
    positionals = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--train", "--predict"):
            if mode is not None:
                raise UsageError("choose either --train or --predict, only once")
            mode = arg[2:]
        elif arg == "--save":
            if savefile is not None:
                raise UsageError("--save given more than once")
            if i + 1 >= len(args):
                raise UsageError("--save needs a SAVEFILE")
            savefile = args[i + 1]
            i += 1
        elif arg.startswith("-"):
            raise UsageError(f"unknown option: {arg}")
        else:
            positionals.append(arg)
        i += 1

    if mode is None:
        raise UsageError("missing --train or --predict")
    if savefile is not None and mode != "train":
        raise UsageError("--save only works with --train")
    if len(positionals) != 2:
        raise UsageError("expected exactly LOADFILE and CHESSFILE")

    loadfile, chessfile = positionals
    for path in (loadfile, chessfile):
        if not os.path.isfile(path):
            raise UsageError(f"{path}: no such file")

    return Options(mode, loadfile, chessfile, savefile)


def main(args):
    """Run the program and return its exit code (0 if OK, 84 on error)."""
    if args in (["-h"], ["--help"]):
        print(HELP)
        return EXIT_OK

    try:
        options = parse_args(args)

    except UsageError as error:
        print(f"my_torch_analyzer: {error}", file=sys.stderr)
        print("Try './my_torch_analyzer --help' for more information.", file=sys.stderr)
        return EXIT_ERROR

    try:
        if options.mode == "predict":
            run_predict(options.loadfile, options.chessfile)
        else:
            run_train(options.loadfile, options.chessfile, options.savefile)
    except (ValueError, OSError) as error:
        print(f"my_torch_analyzer: {error}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK
