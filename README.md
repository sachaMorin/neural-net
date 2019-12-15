# neural-net
Neural net API implemented from scratch using NumPy.
Currently supports fully connected and softmax layers. I'll maybe add convolutional layers at some point in the future. 

## Prerequisites

- Python 3
- Numpy
- [data_fetcher package](https://github.com/sachaMorin/dataset_fetcher)

## Benchmarks
The library was tested on the following datasets:
- [MNIST](http://yann.lecun.com/exdb/mnist/index.html)
- [KMNIST](https://github.com/rois-codh/kmnist)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Banknote Authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- [Sonar (Mines vs. Rocks)](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))

Here's a sample of some "raw" results achieved with this library, with no particular fine-tuning. Keep in mind this is only using a fully connected model :

Dataset      |Classes| L2 Factor | Test Accuracy | Train Accuracy
-------------|---:|-------:|---------:|---:
MNIST        | 10|0.5 |  97.60 %  | 98.64 %
KMNIST       | 10|0.5 |  87.57 %  | 98.07 %
Fashion-MNIST| 10|0.5 |  88.42 %  | 92.23 %
Banknote     | 2|0.0 |  99.27 %  | 99.27 %
Sonar        | 2|0.1 |  85.37 % | 87.95 %

All the above results are the lowest test error achieved over 200 epochs with a learning rate of 0.01, SGD and this architecture :
```python
# MODEL
net = NeuralNetwork(input_size=x_train.shape[1]) # Number of features as input size
net.add_layer(LinearRelu(size=50))
net.add_layer(LinearRelu(size=25))
net.add_layer(LinearSoftmaxCE(size=y_train.shape[1])) # Number of classes as output size

```

## Usage
The data_fetcher repo should be cloned in the working directory. The package can be used to download various datasets and dump them to a pickle file. 

Labels should be one-hot encoded and can be retrieved like so:
```python
from dataset_fetcher.loader import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('MNIST', one_hot=True) # Or KMNIST, Fashion-MNIST, Banknote or Sonar
```

Here's how you can initialize a model, an optimizer and needed data loaders :
```python
from cost import cross_entropy
from data_loader import DataLoader
from optimizers import SGD
from nn import NeuralNetwork
from layers import LinearRelu
from layers import LinearSoftmaxCE

# MODEL
net = NeuralNetwork(input_size=784)
net.add_layer(LinearRelu(size=20))
net.add_layer(LinearSoftmaxCE(size=10) # Size of last layer should match number of classes

# OPTIMIZER
optimizer = SGD(lr=0.01, batch_size=32, weight_decay=0)

# LOADERS
train_loader = DataLoader(x=x_train, y=y_train, batch_size=32,
                          shuffle=True, normalize=True)
test_loader = DataLoader(x=x_test, y=y_test, batch_size=32,
                         shuffle=False, normalize=True,
                         mean=train_loader.get_inputs_mean(),
                         sd=train_loader.get_inputs_sd())
```

A training loop would look like this :
```python
for epoch in range(100):
    for data in enumerate(train_loader):
        # Training
        x_batch, y_batch = data

        net.load_data(x_batch, y_batch)
        net.feedforward()
        net.backpropagation()

        optimizer.step(net)  # Update weights
```
And accuracy can finally be computed over a given set of inputs and labels :
```python
test_acc = net.accuracy(x_test, y_test)
```

`utils.py` also defines a few handy classes to keep track of error and loss as well as draw plots and print results. Have a look at 
`main.py` for examples.
