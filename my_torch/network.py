"""
Neural network
"""


class Network:
    """A chain of layers, called one after the other.

    Example:
        network = Network([Linear(769, 256), ReLU(), Linear(256, 5), Softmax()])
        
        probabilities = network.forward(boards)

    Args:
        layers: The layers of the network, from the first to the last.

    Attributes:
        layers: The list of layers.

    Raises:
        ValueError: If the list of layers is empty.
    """

    def __init__(self, layers):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("A network needs at least one layer")

    def forward(self, inputs):
        """Pass the inputs through every layer, from the first to the last.

        Args:
            inputs: Array of shape (batch, inputs of the first layer).

        Returns:
            The outputs of the last layer.
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
