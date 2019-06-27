import torch

# Create X and track computation
x = torch.ones(2, 2, requires_grad=True)
print(x)

# Do a tensor op
y = x + 2
print(y)

# Check function that created the tensor
print("\n", y.grad_fn)

# play more!
z = y * y * 3
out = z.mean()
print("\n", z, out)

# change requires_flag in place
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print("\n", a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


# Now lets backprop 'out'
out.backward()
print(x.grad)


# Vector Jacobian product 
print("\nVector Jacobian")
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

# get Vector jacobian product
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# Stop autograd:
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)



