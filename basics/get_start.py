#!/usr/bin/env python3
from __future__ import print_function
import torch
import numpy as np

# Create 3x5 unitialized array in pyTorch
x = torch.empty(5, 3)
print(x)

# Randomly initialized matrix
x = torch.rand(3, 5)
print("\n", x)

# Create the zero matrix of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print("\n", x)

# Construct a tensor directly
x = torch.tensor([5.5, 3])
print("\n", x)

# Create a tensor based on existing tensors
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print("\n", x)

# or create a tensor based on an existing tensor.
# These methods will reuse properties of the input tensor
# e.g. dtype, unless new values are provided by user
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print("\n", x)

print("\n Get size of tensor: ", x.size())

# Addition syntax
y = torch.rand(5, 3)
print("\n Add two tensors: \n", x + y)

# Addition syntax 2
print("\n Add two tensors: \n", torch.add(x, y))

# Providing an output tensor as arg for addition
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("\n Output tensor: \n", result)

# MUTATION: adds x to y
y.add_(x)
print("\n In place addition: \n", y)

# Standard numpy syntax
print("\n Numpy syntax: \n", x[:, 1])

# Reshape the tensor
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print("\n Print sizes of resized: \n", x.size(), y.size(), z.size())

# One element tensor - Item to scalar
x = torch.randn(1)
print("\nx:", x)
print("As item:", x.item())

# Convert torch tensor to numpy array
a = torch.ones(5)
b = a.numpy()
print("\nFrom tensor to numpy:", b)

# See numpy v. tensora.add_(1)
a.add_(1)
print("\n", a)
print("Numpy: ", b)

# From numpy to Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print("\nOriginal numpy:", a)
print("New Tensor:", b)

if torch.cuda.is_available():
    print("\nGPU Work")
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
