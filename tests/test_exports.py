"""
Library exports tests
"""

import numpy as np
import pytest

import my_torch
import my_torch.activations
import my_torch.layers
import my_torch.losses
import my_torch.network
import my_torch.optimizers

PUBLIC_CLASSES = {
    "Adam": my_torch.optimizers,
    "CrossEntropyLoss": my_torch.losses,
    "Layer": my_torch.layers,
    "Linear": my_torch.layers,
    "Loss": my_torch.losses,
    "MSELoss": my_torch.losses,
    "Network": my_torch.network,
    "Optimizer": my_torch.optimizers,
    "ReLU": my_torch.activations,
    "SGD": my_torch.optimizers,
    "Sigmoid": my_torch.activations,
    "Softmax": my_torch.activations,
}


def test_the_whole_library_can_be_imported_in_one_line():
    from my_torch import (
        SGD, Adam, CrossEntropyLoss, Layer, Linear, Loss, MSELoss,
        Network, Optimizer, ReLU, Sigmoid, Softmax,
    )


@pytest.mark.parametrize("name", PUBLIC_CLASSES)
def test_each_class_is_the_same_as_in_its_own_folder(name):
    assert getattr(my_torch, name) is getattr(PUBLIC_CLASSES[name], name)


def test_all_lists_exactly_the_public_classes():
    assert sorted(my_torch.__all__) == sorted(PUBLIC_CLASSES)


def test_xor_can_be_learned_with_my_torch_as_mt():
    import my_torch as mt

    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = np.eye(2)[[0, 1, 1, 0]]
    network = mt.Network([mt.Linear(2, 4, seed=0), mt.Sigmoid(),
                          mt.Linear(4, 2, seed=1000), mt.Softmax()])

    network.train(inputs, targets, mt.CrossEntropyLoss(), mt.SGD(network, 0.5),
                  epochs=2000, batch_size=4, seed=0)

    np.testing.assert_array_equal(network.predict(inputs), [0, 1, 1, 0])
