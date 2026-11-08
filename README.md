# my_torch

A neural network library written from scratch in Python with NumPy, and a chess analyzer built with it (Epitech project G-CNA-500).

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Results](#results)
4. [Documentation](#documentation)

## What it does

The analyzer reads chessboards and answers, for each one: `Nothing`, `Check White`, `Check Black`, `Checkmate White` or `Checkmate Black`.

It does not know the rules of chess. A neural network learned the answers from about 2 million example chessboards (572,443 real ones, also shown in a mirror and with their colours swapped).

- **Pushed network**: `my_torch_network.nn`, **92.1%** of right answers on chessboards it has never seen.
- **Made with**: Python and NumPy only. No PyTorch, no TensorFlow.

## Quick start

```bash
make
./my_torch_analyzer --predict my_torch_network.nn boards.txt
```

```
$ cat boards.txt
8/5p1k/R5p1/1p2Q2p/1P2P1qK/6P1/P1r4P/8 w - - 1 45
4B3/8/3k4/5n1P/8/2K2b2/8/8 b - - 2 61
$ ./my_torch_analyzer --predict my_torch_network.nn boards.txt
Checkmate Black
Nothing
```

To create and train a network:

```bash
./my_torch_generator configs/chess_network.conf 1
./my_torch_analyzer --train --save trained.nn chess_network_1.nn train.txt
```

## Results

| Version | Change | Score |
|---|---|---|
| v0.2.0 | first network | 62.4% |
| v0.2.1 | + better starting weights | 79.1% |
| v0.2.2 | + Adam, bigger network | 83.8% |
| v0.2.3 | + early stopping, much less learning by heart | 83.3% |
| v0.3.0 | + mirrored chessboards, learning by heart halved | 87.0% |
| v0.4.0 | + swapped colours, wider network | **92.1%** |

How these choices were measured, with graphs: [Results](docs/results.md).

## Documentation

| Page | Content |
|---|---|
| [Usage](docs/usage.md) | the programs, configuration files, errors, how to train the network again, tests |
| [How it works](docs/how_it_works.md) | from a chessboard to an answer, and how the network learns, in simple words |
| [Library](docs/library.md) | how to use `my_torch` in Python |
| [Results](docs/results.md) | how to read the graphs, and why each setting was chosen |
| [Demo](docs/usage.md#10-demo) | scripts that show the project step by step, for the defense |
