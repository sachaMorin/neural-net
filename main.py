"""Train a neural network on MNIST or MNIST-like datasets.

Implemented from scratch using numpy.
Designed to work on the MNIST dataset and all drop-in replacements
(KMNIST, FashionMNIST etc.).
"""
__author__ = 'Sacha Morin'

from data_loader import DataLoader
from utils import Board
from utils import demo
from dataset_fetcher.loader import load_dataset

from cost import cross_entropy
from optimizers import SGD
from nn import NeuralNetwork
from layers import LinearRelu
from layers import LinearSoftmaxCE

# OPTIONS
# Global options
DATASET = "MNIST"
DEMO = 0  # Number of examples to show after training

# Training options and hyperparameters
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
LAMBDA = 0.5  # L2 Regularization

# Load dataset
x_train, y_train, x_test, y_test, classes = load_dataset(DATASET, one_hot=True)

# MODEL
net = NeuralNetwork(input_size=x_train.shape[1])
net.add_layer(LinearRelu(size=50))
net.add_layer(LinearRelu(size=25))
net.add_layer(LinearSoftmaxCE(size=y_train.shape[1]))

# OPTIMIZER
optimizer = SGD(lr=LR, batch_size=BATCH_SIZE, lamb=LAMBDA)

# INITIALIZATION
# Dataset
print("Loading and processing {}...".format(DATASET))

# Loaders and normalization. Normalize test set with train mean
# and standard deviation
train_loader = DataLoader(x=x_train, y=y_train, batch_size=BATCH_SIZE,
                          shuffle=True, normalize=True)
test_loader = DataLoader(x=x_test, y=y_test, batch_size=BATCH_SIZE,
                         shuffle=False, normalize=True,
                         mean=train_loader.get_inputs_mean(),
                         sd=train_loader.get_inputs_sd())

# TRAINING
print("\nTraining...")
board = Board(DATASET, EPOCHS, net, cross_entropy, train_loader, test_loader,
              LAMBDA)

# Training loop
board(0)
for epoch in range(1, EPOCHS + 1):
    for i, data in enumerate(train_loader, 1):
        # Training
        x_batch, y_batch = data

        net.load_data(x_batch, y_batch)
        net.feedforward()
        net.backpropagation()

        optimizer.step(net)  # Update weights

    # Print progression after epoch
    board(epoch)
    # board.plot_error()

print('\n\nMinimum test error:')
board.print_record()

# Show plots
# DEMO
if DEMO > 0:
    # Display images and predictions
    print("\nDemo...")
    raw_x, x, y = test_loader.get_random_sample(size=DEMO)
    demo(net, classes, raw_x, x, y)
