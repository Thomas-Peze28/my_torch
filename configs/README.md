# Network configuration files

A configuration file describes a network to create with `my_torch_generator`.
It is a JSON file with two keys:

```json
{
  "layers": [
    {"type": "Linear", "input_size": 769, "output_size": 64, "init": "he"},
    {"type": "ReLU"},
    {"type": "Linear", "input_size": 64, "output_size": 5, "init": "he"},
    {"type": "Softmax"}
  ],
  "settings": {
    "loss": "CrossEntropyLoss",
    "optimizer": "SGD",
    "learning_rate": 0.1,
    "epochs": 10,
    "batch_size": 64,
    "seed": 42
  }
}
```

## `layers` (required)

The layers of the network, from the first to the last. Each layer is an object with a `type`:

| `type` | Other keys | What it does |
|---|---|---|
| `Linear` | `input_size`, `output_size` (positive whole numbers), `init` (optional) | a layer of neurons with weights and biases |
| `ReLU` | none | keeps positive numbers, turns negative ones into 0 |
| `Sigmoid` | none | squashes every number between 0 and 1 |
| `Softmax` | none | turns the outputs into probabilities adding up to 1 |

The sizes must follow each other: the `input_size` of a `Linear` layer is the `output_size` of the `Linear` layer before it.

`init` chooses how the starting weights of a `Linear` layer are drawn. The more inputs a neuron has, the smaller its weights must be, so that its total is neither too big nor too small:

| `init` | Starting weights | Made for |
|---|---|---|
| `uniform` (default) | between -1 and 1, whatever the size | very small layers |
| `lecun` | normal, spread `sqrt(1 / input_size)` | any layer |
| `xavier` | uniform between ±`sqrt(6 / (input_size + output_size))` | `Sigmoid` |
| `he` | normal, spread `sqrt(2 / input_size)` | `ReLU` |
For the chess analyzer, the first layer takes 769 inputs (one chessboard) and the last `Linear` layer has 5 outputs (one per answer), followed by `Softmax`.

## `settings` (optional)

How to train the network. They are copied as they are into the generated network file, and read by `my_torch_analyzer --train`:

| Key | Meaning |
|---|---|
| `loss` | `CrossEntropyLoss` or `MSELoss` |
| `optimizer` | `SGD` (same step for every value) or `Adam` (a step adapted to each value) |
| `learning_rate` | size of each correction: about `0.1` for `SGD`, `0.001` for `Adam` |
| `epochs` | number of passes over all the chessboards |
| `batch_size` | number of chessboards in each mini-batch |
| `seed` | makes the generated networks and the training the same every time |

## Files

- `basic_network.conf`: a small network, quick to train.
- `chess_network.conf`: the network planned for the chess analyzer.
