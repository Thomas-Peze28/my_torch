# Usage

[← README](../README.md)

1. [Install](#1-install)
2. [Predict](#2-predict)
3. [Train](#3-train)
4. [Create networks](#4-create-networks)
5. [Configuration files](#5-configuration-files)
6. [Chessboard files](#6-chessboard-files)
7. [Errors](#7-errors)
8. [Make the pushed network again](#8-make-the-pushed-network-again)
9. [Tests](#9-tests)

## 1. Install

The programs need Python 3 and NumPy.

```bash
make
```

For development, use the same versions as the grader (Python 3.14, NumPy 2.3.5, matplotlib 3.10.7):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
source .venv/bin/activate
```

## 2. Predict

```bash
./my_torch_analyzer --predict my_torch_network.nn boards.txt
```

The analyzer prints one answer per chessboard, in the order of the file:

```
Checkmate Black
Nothing
Check White
```

The 5 answers are `Nothing`, `Check White`, `Check Black`, `Checkmate White` and `Checkmate Black`. "White" means that White gives the check.

## 3. Train

```bash
./my_torch_analyzer --train --save trained.nn network.nn train.txt
```

- Every line of `train.txt` needs its expected answer.
- Without `--save`, the trained network replaces `network.nn`.
- The training settings (optimizer, number of epochs...) are stored in the network file. They come from its [configuration file](#5-configuration-files).

10% of the chessboards are put aside and never learned. After every epoch, the network is checked on them. It keeps its best epoch and stops after 3 epochs without progress. It prints one line per epoch:

```
Epoch 10/30: loss 0.1696, accuracy 93.2% | validation loss 0.4970, accuracy 83.9% (best)
```

| Word | Meaning |
|---|---|
| `loss`, `accuracy` | on the chessboards it learns from |
| `validation loss`, `validation accuracy` | on the chessboards put aside |
| `(best)` | the best epoch so far: the one that will be saved |

What loss and accuracy are: see [Results](results.md#2-how-to-read-the-graphs).

## 4. Create networks

```bash
./my_torch_generator configs/basic_network.conf 3
```

Creates 3 untrained networks: `basic_network_1.nn`, `basic_network_2.nn`, `basic_network_3.nn`. Several configurations can be given at once: `./my_torch_generator a.conf 2 b.conf 1`.

## 5. Configuration files

A configuration file is JSON. It describes the layers of a network and how to train it:

```json
{
  "layers": [
    {"type": "Linear", "input_size": 769, "output_size": 64, "init": "he"},
    {"type": "ReLU"},
    {"type": "Linear", "input_size": 64, "output_size": 5, "init": "he"},
    {"type": "Softmax"}
  ],
  "settings": {"optimizer": "Adam", "learning_rate": 0.001, "epochs": 10, "seed": 42}
}
```

**Layers**

| `type` | Keys | Role |
|---|---|---|
| `Linear` | `input_size`, `output_size`, `init` | a layer of neurons |
| `ReLU` | | keeps positive numbers, turns negative ones into 0 |
| `Sigmoid` | | squashes numbers between 0 and 1 |
| `Softmax` | | turns the outputs into probabilities |

- Sizes must follow each other. For chess: 769 inputs first, 5 outputs last, then `Softmax`.
- `init` chooses the starting weights: `uniform` (default), `lecun`, `xavier` or `he`. Use `he` with `ReLU`.

**Settings** (all optional)

| Key | Default | Meaning |
|---|---|---|
| `loss` | `CrossEntropyLoss` | how the error is measured (or `MSELoss`) |
| `optimizer` | `SGD` | how the weights are corrected (or `Adam`) |
| `learning_rate` | `0.1` | size of a correction: about `0.1` for SGD, `0.001` for Adam |
| `epochs` | `10` | maximum number of passes over the chessboards |
| `batch_size` | `64` | chessboards learned before each correction |
| `seed` | none | the same seed gives the same network every time |
| `validation_split` | `0.1` | share of the chessboards put aside (`0` = none) |
| `patience` | `3` | epochs without progress before stopping |

**Files of the repository**

- `configs/chess_network.conf`: the pushed network (769 → 256 → 64 → 5).
- `configs/basic_network.conf`: a small network, quick to train.
- `configs/benchmarks/`: the networks compared in [Results](results.md).

## 6. Chessboard files

One chessboard per line, in FEN, optionally followed by its answer:

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
```

- The FEN has 6 fields. Only the first two are used: the pieces, and the side to move (`w` or `b`).
- The answer is ignored by `--predict` and required by `--train`.
- Empty lines are skipped.

## 7. Errors

On an error, the program writes one message on the error output and exits with code **84**. Nothing is predicted or saved.

| Problem | Example of message |
|---|---|
| wrong arguments | `missing --train or --predict` |
| missing file | `nothing.nn: no such file` |
| broken network file | `chess.nn: not a my_torch network file` |
| network not made for chess | `this network does not take a chessboard ...` |
| broken chessboard | `boards.txt line 2: invalid FEN ...` |
| missing or unknown answer | `train.txt line 1: unknown answer "Check Purple"` |
| impossible place to save | `cannot save the network to missing/new.nn: the folder does not exist` |

## 8. Make the pushed network again

The dataset is not in the repository. Put its folders `check/`, `checkmate/` and `nothing/` in `dataset/`, then:

```bash
make dataset     # 5 seconds
make train       # about 7 minutes
make benchmark   # about 10 minutes
```

| Rule | What it does | Writes |
|---|---|---|
| `make dataset` | removes duplicates, shuffles, cuts into 3 files | `dataset/train.txt` (80%), `validation.txt` (10%), `test.txt` (10%) |
| `make train` | trains `configs/chess_network.conf`, draws its graphs | `my_torch_network.nn`, `benchmarks/pushed_network*` |
| `make benchmark` | trains the networks of `configs/benchmarks/`, draws the comparisons | `benchmarks/init.png`, `benchmarks/optimizer.png` |

The seeds are fixed: the same commands give the same files, byte for byte.

The scripts behind the rules are in `scripts/`. Each one prints its options with `--help`, for example `python3 -m scripts.train_network --help`.

## 9. Tests

```bash
make tests_run
```

The tests (pytest) cover every part: layers on examples computed by hand, the library learning XOR, save and load, the programs run for real on broken files, the scripts on a tiny dataset. The CI runs them on every pull request, in the grader's Docker image.
