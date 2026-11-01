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
