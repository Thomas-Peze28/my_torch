# How it works

[← README](../README.md)

1. [A chessboard becomes numbers](#1-a-chessboard-becomes-numbers)
2. [The network answers](#2-the-network-answers)
3. [The network learns](#3-the-network-learns)
4. [Learning by heart](#4-learning-by-heart)
5. [The three sets of chessboards](#5-the-three-sets-of-chessboards)
6. [How the code is organized](#6-how-the-code-is-organized)

## 1. A chessboard becomes numbers

A network only understands numbers. Each chessboard becomes **769 numbers**, each 0 or 1:

- **768 = 12 × 64**: one grid of 64 squares per piece type (6 White, 6 Black). A square is 1 if that piece is on it.
- **1 more**: 1 if White plays, 0 if Black plays.

The network is not told where the pieces attack. It must learn what a check is from examples.

## 2. The network answers

The pushed network has 4 layers of numbers: **769 → 256 → 64 → 5**.

- A **neuron** adds up its inputs, each multiplied by a **weight**. The weights are what the network learns.
- A **layer** is a group of neurons that read the same inputs.
- Between layers, **ReLU** turns negative numbers into 0. Without it, many layers would act like one.
- The last layer has 5 neurons, one per answer. **Softmax** turns them into 5 probabilities.

The answer is the most likely one:

```
Nothing 3%   Check White 91%   Check Black 1%   Checkmate White 5%   Checkmate Black 0%
→ Check White
```

## 3. The network learns

The network learns from 515,000 chessboards whose answer is known, 64 at a time:

1. **Answer**: it gives its 5 probabilities (forward pass).
2. **Measure the error**: the **loss** is small when it gives a high probability to the right answer, and very big when it is sure of a wrong one.
3. **Share the blame**: going back from the last layer to the first, each weight learns how much it caused the error (backward pass, or backpropagation).
4. **Correct**: each weight moves a little to make the error smaller. **Adam**, the optimizer used, gives each weight its own step size.

One pass over all the chessboards is an **epoch**. The network does up to 30 epochs.

Before the first epoch, the weights are random. They are drawn smaller when a neuron has more inputs (**He** method), so that the totals are not too big.

## 4. Learning by heart

After a few epochs, the network starts to **remember** its chessboards instead of **understanding** them. It gets better on them, but not on new ones. This is called **overfitting**.

To see it, 10% of the chessboards are **put aside**. They are never learned, only used as a test after every epoch. The training keeps the epoch with the best score on them, and stops when the score stops improving (**early stopping**).

The graphs of [Results](results.md) show it.

## 5. The three sets of chessboards

The dataset is cut in three. No chessboard is in two parts.

| Part | Share | Role | Like |
|---|---|---|---|
| train | 80% | learning | exercises |
| validation | 10% | comparing settings | mock exams |
| test | 10% | the final mark, once | the real exam |

## 6. How the code is organized

```
my_torch_analyzer  →  analyzer/  →  my_torch/  →  numpy
```

| Part | Role |
|---|---|
| `analyzer/` | chess: read a FEN, turn it into 769 numbers, turn the answer into text |
| `my_torch/` | the neural network library. It knows nothing about chess. See [Library](library.md). |
| `generator/` | `my_torch_generator` |
| `scripts/` | prepare the dataset, train, draw the graphs |
