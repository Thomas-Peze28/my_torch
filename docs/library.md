# Library

[← README](../README.md)

1. [Example](#1-example)
2. [Network](#2-network)
3. [Layers](#3-layers)
4. [Losses](#4-losses)
5. [Optimizers](#5-optimizers)
6. [Network files](#6-network-files)

`my_torch` is a small neural network library, in the style of PyTorch, written with NumPy only. It knows nothing about chess. The ideas behind it are in [How it works](how_it_works.md).

## 1. Example

A network that learns XOR (1 when the two inputs are different):

```python
import numpy as np
from my_torch import Adam, CrossEntropyLoss, Linear, Network, ReLU, Softmax

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)

network = Network([Linear(2, 8, seed=1, init="he"), ReLU(), Linear(8, 2, seed=2, init="he"), Softmax()])
network.train(inputs, targets, CrossEntropyLoss(), Adam(network, learning_rate=0.01),
              epochs=1000, batch_size=4, seed=0)

print(network.predict(inputs))      # [0 1 1 0]
network.save("xor.nn")
same_network = Network.load("xor.nn")
```

- Arrays have **one row per example**.
- A target row has a 1 for the right answer and 0 elsewhere.
- Run Python from the root of the repository.

## 2. Network

```python
network = Network(layers, settings=None)
```

| Method | Returns |
|---|---|
| `train(inputs, targets, loss, optimizer, epochs, batch_size, seed)` | the history of the training |
| `predict(inputs)` | the index of the most likely answer, for each example |
| `forward(inputs)` | the outputs of the last layer |
| `evaluate(inputs, targets, loss)` | `(loss, accuracy)` on examples, without learning |
| `save(path)` / `Network.load(path)` | write / read a network file |
| `Network.from_config(description)` | a new network from a [configuration](usage.md#5-configuration-files) |

`train` has 3 more optional arguments:

- `validation=(inputs, targets)`: examples never learned, checked after every epoch. The network keeps its best epoch.
- `patience=3`: with `validation`, stop after 3 epochs without progress.
- `verbose=True`: print one line per epoch.

The history has one value per epoch:

```python
{"loss": [...], "accuracy": [...], "val_loss": [...], "val_accuracy": [...], "best_epoch": 7}
```

## 3. Layers

| Class | Role |
|---|---|
| `Linear(input_size, output_size, seed=None, init="uniform")` | a layer of neurons, with weights and biases |
| `ReLU()` | keeps positive numbers, turns negative ones into 0 |
| `Sigmoid()` | squashes numbers between 0 and 1 |
| `Softmax()` | turns each row into probabilities |

`init` chooses the starting weights of `Linear`: `"uniform"`, `"lecun"`, `"xavier"` or `"he"` (best with `ReLU`).

To make a new layer, write a subclass of `Layer` with two methods:

- `forward(inputs)`: compute the outputs;
- `backward(output_error)`: receive the error of the outputs, return the error of the inputs.

## 4. Losses

| Class | Use |
|---|---|
| `CrossEntropyLoss()` | choosing an answer, after `Softmax` |
| `MSELoss()` | predicting numbers |

## 5. Optimizers

| Class | Correction |
|---|---|
| `SGD(network, learning_rate)` | the same step for every weight |
| `Adam(network, learning_rate=0.001)` | a step adapted to each weight |

`train` uses them for you. By hand, one step of learning is:

```python
predictions = network.forward(batch_inputs)
loss.forward(predictions, batch_targets)
network.backward(loss.backward())
optimizer.step()
```

## 6. Network files

`save` writes a compressed NumPy archive. It holds the description of the layers and the settings (JSON), and the arrays of weights.

`load` checks the file and refuses to run any code it may hold. A broken file raises a `ValueError` with the reason.
