# Scripts

Used by `make dataset`, `make train` and `make benchmark`: see [docs/usage.md](../docs/usage.md#8-make-the-pushed-network-again). Each script prints its options with `--help`.

| Script | Role |
|---|---|
| `prepare_dataset.py` | remove duplicates, shuffle and cut the dataset in 3 files; add the training chessboards in a mirror and with their colours swapped |
| `train_network.py` | create, train and measure a network |
| `plot_history.py` | draw learning curves |
| `evaluate_network.py` | accuracy for each answer, confusion matrix |
