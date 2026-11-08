# Results

[← README](../README.md)

1. [Summary](#1-summary)
2. [How to read the graphs](#2-how-to-read-the-graphs)
3. [The pushed network](#3-the-pushed-network)
4. [Starting weights](#4-starting-weights)
5. [Optimizer](#5-optimizer)
6. [Network size](#6-network-size)
7. [Mirrored chessboards](#7-mirrored-chessboards)
8. [Mistakes](#8-mistakes)

Every setting of the network was chosen by measuring it. All graphs can be made again with `make train` and `make benchmark` ([Usage](usage.md#8-make-the-pushed-network-again)).

## 1. Summary

Share of right answers on chessboards the network has never seen:

| Version | Change | Score |
|---|---|---|
| v0.2.0 | first network | 62.4% |
| v0.2.1 | + better starting weights (He) | 79.1% |
| v0.2.2 | + Adam, bigger network | 83.8% |
| v0.2.3 | + early stopping | 83.3% |
| v0.3.0 | + mirrored chessboards | **87.0%** |

Early stopping does not raise the score, but the network learns less by heart (see [3](#3-the-pushed-network)). The mirrored chessboards raise it and halve the learning by heart (see [7](#7-mirrored-chessboards)).

## 2. How to read the graphs

**Epoch** (horizontal axis): one pass of the network over all its chessboards. The graphs show how it changes while it learns.

**Accuracy** (right graph): the share of right answers. **Higher is better.**

**Loss** (left graph): a finer measure of the error. It also counts how sure the network was:

| The right answer is Check White, the network says | Right? | Loss |
|---|---|---|
| Check White at 95% | yes | very small (0.05) |
| Check White at 40% | yes | medium (0.9) |
| Nothing at 90%, Check White at 2% | no | very big (3.9) |

**Lower is better.** The network learns by making its loss smaller.

**Lines**

| Line | Chessboards | Like |
|---|---|---|
| dashed: train | those the network learns from | marks on exercises already done |
| solid: validation | those put aside, never learned | marks at mock exams |

The solid line is what matters: it shows how the network does on new chessboards. **The dot** is the epoch that is kept.

On comparison graphs, **one colour is one network**. The networks are the same except for one setting: the best one is the colour whose solid line ends highest on the right graph.

## 3. The pushed network

![Learning curves of my_torch_network.nn](../benchmarks/pushed_network.png)

- **Right graph**: the dashed line climbs to 93%, but the solid line flattens around 87% after epoch 8. The gap between them is the network **learning by heart**.
- **Left graph**: the solid loss goes slowly back up after epoch 7. The network gets too sure of itself on chessboards it does not know.
- **The dot**: epoch 12 has the best score on the chessboards put aside (87.1%), so it is kept. The training stopped 3 epochs later.

The best epoch is chosen by **accuracy**, not loss: the grader counts right answers. In v0.2.3, chosen by loss, epoch 4 would have been kept, with 82.6% instead of 83.5%.

Early stopping, measured in v0.2.3: compared with 20 epochs without it, the gap between train and validation went down from 13.7 to 10.8 points, for the same score.

The 10% put aside during training come from `train.txt`, so they can hold the mirror of a learned chessboard: the solid line is a little optimistic. The scores of this page are measured on `validation.txt` and `test.txt`, which have no mirror.

## 4. Starting weights

Four identical small networks, only their starting weights differ:

![Starting weights](../benchmarks/init.png)

| Starting weights | Validation |
|---|---|
| uniform (between -1 and 1) | 61.9% |
| LeCun | 78.7% |
| Xavier | 78.2% |
| **He** | **78.8%** |

- **Green** (uniform) stays far below: its weights are too big, so the totals of the neurons explode and the network learns badly.
- The **three others** scale the weights to the number of inputs: +17 points. They are so close that their lines overlap.
- **He** is chosen: it is made for `ReLU`, and is the best by a little.

## 5. Optimizer

Two identical small networks, one corrected by SGD, the other by Adam:

![Optimizer](../benchmarks/optimizer.png)

| Optimizer | Validation |
|---|---|
| SGD | 78.8% |
| **Adam** | **80.5%** |

- **Blue** (Adam) is above from the first epoch.
- Adam gives each weight its own step size. Many of our weights are rarely used (only about 20 of the 769 inputs are 1 on a chessboard), and Adam pushes them more.
- **Adam** is chosen. It is 1.5 times slower per epoch, and learns by heart a bit faster, which early stopping handles.

## 6. Network size

| Network | Validation |
|---|---|
| 769 → 64 → 5 | 81.4% |
| **769 → 256 → 64 → 5** | **83.6%** |

With He and Adam, the bigger network gains 2 points. With the first settings, it was not better (61.1% against 61.9%): a bigger network only helps once it learns well.

## 7. Mirrored chessboards

A chessboard seen in a mirror (columns a and h swapped, b and g...) has the same answer: a check stays a check, by the same colour. `make dataset` adds the mirror of every training chessboard to `train.txt`, so the network learns from twice more different chessboards (1,141,784 instead of 572,443). Mirrors are never added to `validation.txt` or `test.txt`, and a mirror that is already in one of them is skipped.

Same network (769 → 256 → 64 → 5, He, Adam), with and without the mirrors:

![Mirrored chessboards](../benchmarks/augmentation.png)

| Training chessboards | Learned (kept epoch) | Validation | Gap | Test |
|---|---|---|---|---|
| without mirrors | 93.2% | 83.5% | 9.7 | 83.3% |
| **with mirrors** | 92.6% | **87.0%** | **5.6** | **87.0%** |

- **Orange** (mirrors) is above blue on the right graph at every epoch.
- Its dashed line climbs **less** than blue: it learns less by heart.
- On the left graph, the solid loss of blue goes back up from epoch 4, orange stays almost flat.
- Training takes 9 minutes instead of 7.

The mirrors are kept: +3.7 points on the test, and the gap is almost halved.

## 8. Mistakes

![Confusion matrix of my_torch_network.nn](../benchmarks/pushed_network_confusion.png)

How to read it:

- each **row** is the right answer, each **column** the answer of the network;
- the **diagonal** (dark green) holds the right answers;
- every other cell is a mistake. For example, row Checkmate Black, column Check Black: 13.6% of the Black checkmates were taken for Black checks.

| Answer | Found |
|---|---|
| Nothing | 91.7% |
| Check White | 83.5% |
| Check Black | 81.8% |
| Checkmate White | 85.0% |
| Checkmate Black | 85.0% |

What it shows:

- The network **never mixes up the colours**: the side to move is one of its inputs.
- Its mistakes are telling a **check from a checkmate**, and a **check from nothing**. To see a check, it must see that two far pieces are on the same line with nothing between them: the hardest thing to learn.
