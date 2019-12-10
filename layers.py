"""Layers definition.

Layers should be added to neural network instance via the
NeuralNetwork.add_layer() method with desired size as parameter. ex:
    net = NeuralNetwork()
    net.add_layer(LinearRelu(100))
"""
import numpy as np
import math


class Linear:
    """Linear layer parent class.

    Subclasses should implement activation and activation_grad methods.
    """
    def __init__(self, size):
        self._size = size
        self.weights = None
        self.grad_batch = None
        self._last_input = None  # Holds last input
        self._z = None  # Linear transformation of input
        self._a = None  # Activated linear transformation of input
        self._d = None  # Error

    def initialize(self, previous_size):
        # Should only be called by a NeuralNetwork instance when layer
        # is added to network (see NeuralNetwork.add_layer())
        previous_size += 1  # Account for bias unit
        self.weights = self._init_weights(previous_size, self._size)
        self.grad_batch = np.zeros((previous_size, self._size))

    def _init_weights(self, rows, cols):
        # Random weight initialization
        stdv = 1 / math.sqrt(self._size)
        return np.random.uniform(-stdv, stdv, (rows, cols))

    def get_weights(self):
        return self.weights

    def get_size(self):
        return self._size

    def feedforward(self, input):
        input = np.insert(input, 0, 1, axis=1)  # Add bias weights
        self._last_input = input  # Save input for backprop
        self._z = input @ self.weights  # Linear
        self._a = self.activation(self._z)  # Activate

        return self._a

    def backpropagation(self, d, propagate):
        self._d = d[:, 1:] * self.activation_grad(self._z)  # Ignore bias unit
        self.grad_batch = self._last_input.T @ self._d

        if propagate:
            return self._d @ self.weights.T
        else:
            # Avoid matrix multiplication if not needed
            return None

    def activation(self, z):
        raise NotImplementedError("Linear subclasses should implement "
                                  "an activation method.")

    def activation_grad(self, z):
        raise NotImplementedError("Linear subclasses should implement an "
                                  "activation_grad method to compute "
                                  "layer gradient.")


class LinearRelu(Linear):
    """Linear layer with Relu activation."""
    def activation(self, z):
        # Relu
        z[z < 0] = 0
        return z

    def activation_grad(self, z):
        # Relu gradient
        z[z > 0] = 1
        z[z <= 0] = 0
        return z


class LinearSoftmaxCE(Linear):
    """Linear output layer with softmax activation and cross-entropy gradient.

    Main output layer. Assumes cross-entropy is used as cost function.
    Softmax activation to output a valid probability distribution. Layer size
    should match the number of classes in dataset.
    """
    def activation(self, z):
        # Numerically stable Softmax
        exp_ = np.exp(z - np.max(z))
        return exp_ / np.sum(exp_, axis=1, keepdims=True)

    def activation_grad(self, z):
        # Not needed because backpropagation method is overridden
        pass

    def backpropagation(self, y, propagate):
        self.d = self._a - y
        self.grad_batch = self._last_input.T @ self.d

        if propagate:
            return self.d @ self.weights.T
        else:
            return None
