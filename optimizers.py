"""Optimizer definitions."""


class MiniBatch:
    def __init__(self, lr=0.001, batch_size=32, lamb=0):
        self._lr = lr
        self._batch_size = batch_size
        self._lambda = lamb

    def step(self, nn):
        """Update all weights in a neural network.

        Gradients should be computed first.

        Args:
            nn(nn.NeuralNetwork): Neural net to be updated.

        """
        for layer in nn:
            self.regularize(layer)
            self.step_layer(layer)

    def regularize(self, layer):
        # L2 Regularization. Add weight values to gradient (except bias unit)
        if self._lambda != 0:
            layer.grad_batch[1:] += self._lambda * layer.weights[1:]

    def step_layer(self, layer):
        # Update weights of a single layer
        raise NotImplementedError("Optimizers should implement a step_layer "
                                  "method.")


class SGD(MiniBatch):
    """Stochastic gradient descent."""
    def step_layer(self, layer):
        layer.weights -= self._lr * 1 / self._batch_size * layer.grad_batch
