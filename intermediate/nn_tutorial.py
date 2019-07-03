from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math

# Set up and install data in the correct folder
# Using pathlib and requests to do this
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

# Data is stored as numpy array, and has been stored using pickle.
# Imports gzip and pickle in order to unmarshall and open
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# Reformat to 2x2 array from 1dim array and print it
# Import numpy and and from matplotlib, import pyplot
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# print(x_train.shape)

# Convert from numpy to torch
# Import torch
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
# print(x_train, y_train)
# print(x_train.shape)
# print(y_train.min(), y_train.max())

# Making neural net from scratch without torch.nn module
# Import math
weights = torch.randn(784, 10)/ math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# Any python function or callable object can be a model
# Softmax is teh activation function (instead of ReLu)
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    # The @ symbol is a numpy operator for the dot product
    return log_softmax(xb @ weights + bias)

# One forward pass
bs = 64  # batch size
xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
# print(preds[0], preds.shape)

# Create simple loss function
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

# Check loss for later comparison
yb = y_train[0:bs]
# print(loss_func(preds, yb))

#  For each prediction, if the index with the largest value matches the target value, then the prediction was correct.
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))
