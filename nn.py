"""Main neural network class."""
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size):
        """Initialize an empty net.

        Args:
            input_size (int): Number of features.
        """
        self._input_size = input_size
        self._x = None
        self._y = None
        self._layers = []
        self._output = None

    def add_layer(self, layer):
        """Add layer to model.

        Args:
            layer (Layer): A layer instance. Layers are added to model
            following the call order.

        """
        no_ly = len(self._layers)
        if no_ly == 0:
            layer_input = self._input_size  # First layer
        else:
            layer_input = self._layers[no_ly - 1].get_size()  # Other layers

        layer.initialize(layer_input)  # Initialize weights
        self._layers.append(layer)  # Add layers to model

    def __iter__(self):
        return iter(self._layers)

    def reverse(self):
        return iter(reversed(self._layers))

    def load_data(self, x, y):
        self._y = y
        self.load_input(x)

    def load_input(self, x):
        self._x = x

    def feedforward(self):
        """Forward pass."""
        current = self._x
        for ly in self:
            current = ly.feedforward(current)
        self._output = current  # Set output

    def backpropagation(self):
        """Backward pass to compute gradient."""
        current = self._y
        last = len(self._layers)
        for i, ly in enumerate(self.reverse()):
            current = ly.backpropagation(current, i < last)

    def predict(self, x):
        """Load data, feedforward and return index of predicted class."""
        self.load_input(x)
        self.feedforward()

        return self._output.argmax(axis=1)

    def accuracy(self, x, y):
        """Compute accuracy over provided sample.

        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.

        Returns:
            accuracy(float): Model accuracy (%)
        """
        correct = np.sum(self.predict(x) == y.argmax(axis=1))

        return correct / x.shape[0]
