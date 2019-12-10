"""Cost function definitions."""
import numpy as np


def cross_entropy(nn, x, y, lamb):
    """Compute cross entropy loss on given examples.

    Args:
        nn(nn.NeuralNetwork): Model for prediction.
        x(ndarray): Normalized input.
        y(ndarray): Labels.
        lamb(float): L2 regularization factor.

    Returns:
        loss(float): Cross entropy loss over inputs.

    """
    # Predict
    nn.load_data(x, y)
    nn.feedforward()

    # Compute L2 regularization term
    # Sum squared weights (except bias unit)
    reg = sum([np.sum(ly.get_weights()[1:] ** 2) for ly in nn])

    # Compute cross entropy. Most products being cancelled in standard CE
    # formula, select only output with corresponding label 1 and not 0.
    ce = -np.sum(np.log(nn._output[np.arange(len(y)), y.argmax(axis=1)]))

    # Return average CE + L2 regularization
    return (ce + 0.5 * lamb * reg) / x.shape[0]
