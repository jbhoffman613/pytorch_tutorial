from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn
from IPython.core.debugger import set_trace
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Set up and install data in the correct folder
# Using pathlib and requests to do this
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Set up and organize data for training
# Convert from numpy to torch
# Import torch
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
bs = 64  # batch size
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# Validation is not back prop so we can double the batch size. Needs less memory and computation
valid_ds = TensorDataset(x_valid, y_valid)

# Learning constants
lr = 0.1  # learning rate
epochs = 2  # how many epochs to train for

# Create loss func using built torch function
loss_func = F.cross_entropy

# Subclass and our model func turned into a class
# The body of forward() is the same as our old model func
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

# Model for CNN with 3 conv layers
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

# Simplified method of defining a model - this class is just one layer
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

#  For each prediction, if the index with the largest value matches the target value, then the prediction was correct.
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# print("Old accuracy:", accuracy(preds, yb))

# New model instantiations and fit
def get_model():
    model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
    )
    model.to(dev)
    # Momentum is a variation on stochastic gradient descent
    return model, optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Compute the loss for the training and validation in one step
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# un trained accuracy
# xb = x_train[0:bs]  # a mini-batch from x
# # Check loss for later comparison
# yb = y_train[0:bs]
# print(loss_func(model(xb), yb))

# Train the model and get the necessary losses
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # first train
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        # Then eval
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
