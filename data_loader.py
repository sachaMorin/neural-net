"""Data loader for batch loading with normalization."""
import numpy as np


def _para_shuffle(x, y):
    # Shuffle x and y arrays in unison
    m = x.shape[0]
    shuf = np.arange(m)
    np.random.shuffle(shuf)

    return x[shuf], y[shuf]


class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, normalize=False,
                 mean=None, sd=None):
        """DataLoader for batch iteration.

        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels (one hot encoding).
            batch_size(int): Number of examples to return per batch.
            shuffle(boolean, optional): Whether to shuffle dataset before
            every epoch.
            normalize(boolean, optional):
            mean(float, optional): Mean used for normalization.
            sd(float, optional): Standard deviation used for normalization.
        """

        # Normalization
        if normalize:
            # If no mean and std are given, compute from dataset
            if mean is None:
                mean = np.mean(x)
            if sd is None:
                sd = np.std(x)
            x = (x - mean) / sd  # Normalize
        else:
            # If normalization is set to false, mean and sd should be
            # set to None
            mean = None
            sd = None

        self._normalize = normalize
        self._mean = mean
        self._sd = sd
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._x = x
        self._y = y

    def __iter__(self):
        """Batch iteration."""
        class BatchIterator:
            def __init__(self, x, y, batch_size, shuffle):
                self._x = x
                self._y = y
                self._index = 0
                self._shuffle = shuffle
                self._batch_size = batch_size

            def __next__(self):
                if self._index >= len(self._x):
                    raise StopIteration

                if self._index is 0 and self._shuffle:
                    # Shuffle when starting a new epoch.
                    self._x, self._y = _para_shuffle(self._x, self._y)

                x_batch = self._x[self._index:self._index + self._batch_size]
                y_batch = self._y[self._index:self._index + self._batch_size]
                self._index += self._batch_size

                return x_batch, y_batch

        return BatchIterator(self._x, self._y, self._batch_size, self._shuffle)

    def get_all(self):
        """Return both inputs and labels."""
        return self._x, self._y

    def get_inputs(self):
        """Return all inputs"""
        return self._x

    def get_labels(self):
        """Return all labels."""
        return self._y

    def get_inputs_mean(self):
        """Return mean used for normalization."""
        return self._mean

    def get_inputs_sd(self):
        """Return standard deviation used for normalization."""
        return self._sd

    def get_random_sample(self, size):
        """Get random sample.

        Args:
            size(int): sample size.

        Returns:
            (tuple): containing:
                raw(ndarray): Raw inputs (for display).
                x(ndarray): Normalized inputs.
                y(ndarray): Labels.
        """
        selection = np.random.randint(0, len(self._x), size=size)
        if self._normalize:
            # 'Unnormalize'
            raw = self._x[selection] * self._sd + self._mean
        else:
            raw = self._x[selection]
        return raw, self._x[selection], self._y[selection]
