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

Creates a network from `configs/basic_network.conf` (like `./my_torch_generator`), trains it on `dataset/train.txt` (like `./my_torch_analyzer --train`), saves it as `my_torch_network.nn` and prints its accuracy on `dataset/validation.txt`.

| Option | Default | Meaning |
|---|---|---|
| `--config FILE` | `configs/basic_network.conf` | network and training settings |
| `--dataset FOLDER` | `dataset` | folder with the three files |
| `--output FILE` | `my_torch_network.nn` | trained network |
| `--test` | off | also print the accuracy on `dataset/test.txt` |

The seed of the configuration fixes the starting weights and the order of the mini-batches: the same command gives the same network.
The same network can be made with the two programs:

```bash
./my_torch_generator configs/basic_network.conf 1
./my_torch_analyzer --train --save my_torch_network.nn basic_network_1.nn dataset/train.txt
```

## Results

Share of right answers. The test file is only used for the network that is pushed.

| Network | Config | Layers | Training time | Train | Validation | Test |
|---|---|---|---|---|---|---|
| `my_torch_network.nn` (v0.2.1) | `basic_network.conf` (`he` weights) | 769 → 64 → 5 | 2 min | 81.8% | 79.2% | 79.1% |
| v0.2.0 network | `basic_network.conf` of v0.2.0 (`uniform` weights) | 769 → 64 → 5 | 2 min | 62.5% | 61.9% | 62.4% |
| not pushed | `chess_network.conf` of v0.2.0 (`uniform` weights) | 769 → 256 → 64 → 5 | 10 min | 62.4% | 61.1% | - |

With `uniform` weights, train and validation are almost the same: the networks do not learn by heart, they do not learn enough, and the bigger network is not better.
With `he` weights (see below), the same small network gains 17 points. Train is now 2.6 points above validation: the network starts to learn a little by heart, to watch during longer trainings.

### Starting weights

The same network (`basic_network.conf`: 769 → 64 → 5, SGD, learning rate 0.1, 10 epochs, seed 42), trained 4 times with only `init` changed:

| `init` | Loss, first epoch → last | Validation |
|---|---|---|
| `uniform` | 1.113 → 0.787 | 61.9% |
| `lecun` | 0.820 → 0.445 | 78.6% |
| `xavier` | 0.825 → 0.455 | 78.1% |
| `he` | 0.821 → 0.449 | **79.2%** |

Weights scaled to the size of the layer give +17 points. `he`, made for `ReLU`, is the best: the configurations now use it.
