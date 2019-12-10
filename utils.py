"""Utilities."""

from matplotlib import pyplot as plt
import numpy as np


def pretty_print(percent, train_error, train_loss,
                 test_error,
                 test_loss):
    """Formatted print of metrics."""
    print("{:6.2f} % | "
          "train error : {:7.4f} % | "
          "train loss : {:6.4f} | "
          "test error : {:7.4f} % | "
          "test loss : {:6.4f} "
          .format(100 * percent, 100 * train_error,
                  train_loss, 100 * test_error, test_loss))


class PlotTracker:
    """Track data points and plot them.

    Track error and loss on training and test set respectively. Display
    data points on line charts.
    """

    def __init__(self, dataset_name):
        self._dataset_name = dataset_name
        self._epoch_no = []
        self._train_loss = []
        self._test_loss = []
        self._train_error = []
        self._test_error = []

    def add(self, epoch, train_loss, test_loss, train_error, test_error):
        """Save data point."""
        self._epoch_no.append(epoch)
        self._train_loss.append(train_loss)
        self._test_loss.append(test_loss)
        self._train_error.append(train_error * 100)
        self._test_error.append(test_error * 100)

    def plot_loss(self, show=True, save_path=None):
        """Display loss plot."""

        plt.clf()
        plt.plot(self._epoch_no, self._train_loss, label='Training')
        plt.plot(self._epoch_no, self._test_loss, label='Validation')
        plt.title('Model loss on {}'.format(self._dataset_name))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.gca().set_ylim([0, 0.1])  # Fixed scale on y axis
        plt.gca().set_yticks(np.arange(0, 0.11, 0.001))

        if show:
            plt.draw()
            plt.pause(0.001)

        if save_path:
            plt.savefig(save_path)

    def plot_error(self, show=True, save_path=None):
        """Display error plot."""

        plt.clf()
        plt.plot(self._epoch_no, self._train_error, label='Training')
        plt.plot(self._epoch_no, self._test_error, label='Validation')
        plt.title('Model Error on {}'.format(self._dataset_name))
        plt.xlabel('Epochs')
        plt.ylabel('Error (%)')
        plt.legend()
        plt.grid()
        plt.gca().set_ylim([0, 1])  # Fixed scale on y axis
        plt.gca().set_yticks(np.arange(0, 101, 5))

        if show:
            plt.draw()
            plt.pause(1)

        if save_path:
            plt.savefig(save_path)

    def get_record(self, total_epochs):
        test_error_min = min(self._test_error)
        i = self._test_error.index(test_error_min)
        epoch = self._epoch_no[i]
        test_error_min = test_error_min/100
        train_error_min = self._train_error[i]/100
        train_loss_min = self._train_loss[i]
        test_loss_min = self._test_loss[i]

        return epoch/total_epochs, train_error_min, train_loss_min, \
               test_error_min, test_loss_min


class Board:
    def __init__(self, dataset, total_epochs, net, cost, train_loader,
                 test_loader, lamb):
        self._total_epochs = total_epochs
        self._net = net
        self._cost = cost
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._lambda = lamb
        self._plot_tracker = PlotTracker(dataset)

    def __call__(self, epoch):
        """"Helper function to update plots and print metrics."""
        # Error and loss on train set
        train_error = 1 - self._net.accuracy(
            *self._train_loader.get_all())
        train_loss = self._cost(self._net, *self._train_loader.get_all(),
                                self._lambda)

        # Error and loss on test set
        test_error = 1 - self._net.accuracy(*self._test_loader.get_all())
        test_loss = self._cost(self._net, *self._test_loader.get_all(),
                               self._lambda)

        # Add points to plot
        self._plot_tracker.add(epoch, train_loss, test_loss, train_error,
                               test_error)

        # Formatted print
        pretty_print(epoch / self._total_epochs, train_error,
                     train_loss, test_error, test_loss)

    def plot_error(self):
        self._plot_tracker.plot_error()

    def plot_loss(self):
        self._plot_tracker.plot_loss()

    def print_record(self):
       pretty_print(*self._plot_tracker.get_record(self._total_epochs))


def report(epoch, total_epochs, net, cost, train_loader, test_loader,
           weight_decay, PlotTracker):
    """"Helper function to update plots and print metrics."""
    # Error and loss on train set
    train_error = 1 - net.accuracy(*train_loader.get_all())
    train_loss = cost(net, *train_loader.get_all(), weight_decay)

    # Error and loss on test set
    test_error = 1 - net.accuracy(*test_loader.get_all())
    test_loss = cost(net, *test_loader.get_all(), weight_decay)

    # Add points to plot
    PlotTracker.add(epoch, train_loss, test_loss, train_error, test_error)

    # Formatted print
    pretty_print(epoch / total_epochs, train_error,
                 train_loss, test_error, test_loss)


def demo(nn, classes, raw_x, x, y):
    """Shows image, neural net prediction and real label to user.

    Args:
        nn(nn.NeuralNetwork): Model for prediction.
        classes(tuple): Dataset classes.
        raw_x(ndarray): Test examples(unnormalized for proper display).
        x(ndarray): Test examples(normalized for prediction).
        y(ndarray): Test labels.

    """
    predictions = [classes[i] for i in nn.predict(x)]
    labels = [classes[i] for i in y.argmax(axis=1)]
    plt.gray()
    plt.close()
    for i, example in enumerate(raw_x):
        plt.imshow(example.reshape(28, 28))
        plt.title("Predicted : " + predictions[i] + "   |   Label : "
                  + labels[i], fontweight="bold")
        plt.axis("off")
        plt.show()
        # plt.draw()
        # print("Press key to continue...")
        # plt.waitforbuttonpress(0)
        # plt.close()
