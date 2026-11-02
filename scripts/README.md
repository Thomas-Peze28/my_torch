# Scripts

The scripts used to prepare the dataset and train the networks. Run them from the root of the repository, with a Python that has NumPy (`source .venv/bin/activate` on a development machine).

The dataset is not in the repository: put its folders `check/`, `checkmate/` and `nothing/` in `dataset/` first.

## Prepare the dataset

```bash
make dataset        # python3 -m scripts.prepare_dataset
```

Merges the 9 files of `dataset/`, removes the duplicates (same pieces and same side to move), shuffles them with seed 42 and writes:

| File | Share | Used for |
|---|---|---|
| `dataset/train.txt` | 80% | learning |
| `dataset/validation.txt` | 10% | checking the network during and after training |
| `dataset/test.txt` | 10% | the final mark, never to choose a setting |

## Train a network

```bash
make train          # python3 -m scripts.train_network
```

Creates a network from `configs/chess_network.conf` (like `./my_torch_generator`), trains it on `dataset/train.txt` (like `./my_torch_analyzer --train`), saves it as `my_torch_network.nn` and prints its accuracy on `dataset/validation.txt`.

| Option | Default | Meaning |
|---|---|---|
| `--config FILE` | `configs/chess_network.conf` | network and training settings |
| `--dataset FOLDER` | `dataset` | folder with the three files |
| `--output FILE` | `my_torch_network.nn` | trained network |
| `--test` | off | also print the accuracy on `dataset/test.txt` |

The seed of the configuration fixes the starting weights and the order of the mini-batches: the same command gives the same network.
The same network can be made with the two programs:

```bash
./my_torch_generator configs/chess_network.conf 1
./my_torch_analyzer --train --save my_torch_network.nn chess_network_1.nn dataset/train.txt
```

## Results

Share of right answers. The test file is only used for the network that is pushed.

| Network | Layers | Starting weights | Optimizer | Epochs | Time | Train | Validation | Test |
|---|---|---|---|---|---|---|---|---|
| **`my_torch_network.nn` (from v0.2.2)** | 769 → 256 → 64 → 5 | `he` | Adam 0.001 | 20 | 9 min | 97.3% | 83.6% | 83.8% |
| not pushed | 769 → 64 → 5 | `he` | Adam 0.001 | 10 | 3 min | 85.3% | 81.4% | - |
| v0.2.1 network | 769 → 64 → 5 | `he` | SGD 0.1 | 10 | 2 min | 81.8% | 79.2% | 79.1% |
| v0.2.0 network | 769 → 64 → 5 | `uniform` | SGD 0.1 | 10 | 2 min | 62.5% | 61.9% | 62.4% |
| not pushed | 769 → 256 → 64 → 5 | `uniform` | SGD 0.1 | 20 | 10 min | 62.4% | 61.1% | - |

Train is measured on the first 71,556 chessboards of `train.txt`, the size of the validation file.

- With `uniform` weights, train and validation are almost the same: the networks do not learn enough, and the bigger network is not better.
- He weights (see below) give +17 points, Adam (see below) +2 more, and with both the bigger network becomes useful.
- The pushed network learns by heart: 97.3% on the chessboards it has seen, 83.6% on new ones. Fighting this (checking the validation at each epoch, stopping at the best one) is the next step.

### Starting weights

The same network (`basic_network.conf`: 769 → 64 → 5, SGD, learning rate 0.1, 10 epochs, seed 42), trained 4 times with only `init` changed:

| `init` | Loss, first epoch → last | Validation |
|---|---|---|
| `uniform` | 1.113 → 0.787 | 61.9% |
| `lecun` | 0.820 → 0.445 | 78.6% |
| `xavier` | 0.825 → 0.455 | 78.1% |
| `he` | 0.821 → 0.449 | **79.2%** |

Weights scaled to the size of the layer give +17 points. `he`, made for `ReLU`, is the best: the configurations now use it.

### Optimizer

The same network (`basic_network.conf`: 769 → 64 → 5, He weights, 10 epochs, seed 42), trained with only the optimizer changed:

| Optimizer | Loss, first epoch → last | Train | Validation | Check found | Checkmate found | Nothing found |
|---|---|---|---|---|---|---|
| SGD, learning rate 0.1 | 0.821 → 0.449 | 81.8% | 79.2% | 69.1% | 77.3% | 88.7% |
| Adam, learning rate 0.001 | 0.752 → 0.387 | 85.3% | **81.4%** | **74.2%** | **80.4%** | 88.0% |

Adam gives every weight its own step: +2.2 points on validation, +5 points on Check. It also learns by heart faster (train - validation: 3.9 points instead of 2.6), and each epoch is about 1.5 times slower.
