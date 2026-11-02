"""
Validation and early stopping tests
"""

import numpy as np
import pytest

from my_torch import Adam, CrossEntropyLoss, Linear, Network, ReLU, Softmax

LOSS = CrossEntropyLoss()


def random_examples(count, seed):
    """Examples with answers drawn at random: there is nothing to understand,
    so the network can only learn the training ones by heart."""
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(count, 10))
    targets = np.eye(3)[rng.integers(0, 3, size=count)]
    return inputs, targets


def big_network():
    return Network([Linear(10, 64, seed=1, init="he"), ReLU(), Linear(64, 3, seed=2, init="he"), Softmax()])


def train(network, epochs=60, patience=None, verbose=False):
    inputs, targets = random_examples(60, seed=0)
    validation = random_examples(60, seed=1)
    history = network.train(inputs, targets, LOSS, Adam(network, learning_rate=0.01), epochs, 10, seed=0,
                            validation=validation, patience=patience, verbose=verbose)
    return history, validation


def test_validation_is_measured_after_every_epoch():
    history, _ = train(big_network(), epochs=8)
    assert [len(history[key]) for key in ("loss", "accuracy", "val_loss", "val_accuracy")] == [8, 8, 8, 8]
    assert all(0 <= value <= 1 for value in history["accuracy"] + history["val_accuracy"])


def test_learning_by_heart_is_seen_on_the_validation():
    history, _ = train(big_network())
    assert history["accuracy"][-1] > 0.9
    assert history["val_loss"][-1] > min(history["val_loss"])


def test_best_epoch_is_the_highest_validation_accuracy():
    history, _ = train(big_network())
    assert history["best_epoch"] == int(np.argmax(history["val_accuracy"])) + 1


def test_the_network_keeps_the_weights_of_its_best_epoch():
    network = big_network()
    history, (val_inputs, val_targets) = train(network)
    kept = network.forward(val_inputs)
    best = history["best_epoch"] - 1
    assert LOSS.forward(kept, val_targets) == pytest.approx(history["val_loss"][best])
    assert np.mean(kept.argmax(axis=1) == val_targets.argmax(axis=1)) == history["val_accuracy"][best]


def test_training_stops_after_patience_epochs_without_progress():
    history, _ = train(big_network(), epochs=60, patience=3)
    assert len(history["loss"]) == history["best_epoch"] + 3
    assert len(history["loss"]) < 60


def test_without_validation_every_epoch_is_run_and_nothing_is_kept():
    inputs, targets = random_examples(30, seed=0)
    network = big_network()
    history = network.train(inputs, targets, LOSS, Adam(network), 5, 10, seed=0, patience=1)
    assert len(history["loss"]) == 5
    assert history["val_loss"] == [] and history["best_epoch"] is None


def test_verbose_prints_one_line_per_epoch(capsys):
    history, _ = train(big_network(), epochs=60, patience=2, verbose=True)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(history["loss"]) + 1
    assert lines[0].startswith("Epoch 1/60: loss ")
    assert "validation loss" in lines[0] and lines[0].endswith("(best)")
    assert lines[-1] == "No progress for 2 epochs: stopped"


def test_nothing_is_printed_by_default(capsys):
    train(big_network(), epochs=3)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("patience", [0, -2])
def test_patience_must_be_positive(patience):
    network = big_network()
    with pytest.raises(ValueError, match="Patience"):
        network.train(*random_examples(10, 0), LOSS, Adam(network), 2, 5, patience=patience)


def test_evaluate_gives_the_loss_and_the_share_of_right_answers():
    network = Network([Linear(2, 2, seed=0), Softmax()])
    network.layers[0].weights[...] = [[1.0, 0.0], [0.0, 1.0]]
    inputs = np.array([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0]])
    targets = np.eye(2)[[0, 1, 1, 1]]
    val_loss, accuracy = network.evaluate(inputs, targets, LOSS)
    assert accuracy == 0.75
    assert val_loss == pytest.approx(LOSS.forward(network.forward(inputs), targets))
