# %% [markdown]
# # Lab 2 — Backpropagation by hand
#
# A two-layer network in NumPy. We derive the gradients, check them against a
# finite difference, and train XOR. The check is the point: a plausible loss
# curve can hide a wrong derivative.
#
# **Milestone:** the analytic and numerical gradients agree to better than 1e-6.

# %%
import numpy as np

rng = np.random.default_rng(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros((1, 4))
W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X):
    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    return z1, h, z2, sigmoid(z2)

def loss(p, y):
    return float((-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))).mean())

def backward(X, y, z1, h, p):
    d2 = (p - y) / len(X)                 # dL/dz2 for sigmoid + BCE
    gW2 = h.T @ d2
    gb2 = d2.sum(0, keepdims=True)
    d1 = (d2 @ W2.T) * h * (1 - h)        # dL/dz1
    gW1 = X.T @ d1
    gb1 = d1.sum(0, keepdims=True)
    return gW1, gb1, gW2, gb2

z1, h, z2, p = forward(X)
grads = backward(X, y, z1, h, p)
print("loss before training:", round(loss(p, y), 4))
print("gradient shapes:", [g.shape for g in grads])

# %%
# finite-difference check on a few coordinates
eps = 1e-5
checks = []
for M, g, idxs in [(W1, grads[0], [(0, 0), (1, 2)]), (W2, grads[2], [(0, 0), (3, 0)])]:
    for i, j in idxs:
        old = M[i, j]
        M[i, j] = old + eps
        lp = loss(forward(X)[3], y)
        M[i, j] = old - eps
        lm = loss(forward(X)[3], y)
        M[i, j] = old
        checks.append(abs((lp - lm) / (2 * eps) - g[i, j]))
print("max finite-difference error:", f"{max(checks):.2e}")

# %%
for step in range(6000):
    z1, h, z2, p = forward(X)
    gW1, gb1, gW2, gb2 = backward(X, y, z1, h, p)
    lr = 0.5
    W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2

z1, h, z2, p = forward(X)
pred = (p > 0.5).astype(int)
print("final loss:", round(loss(p, y), 5))
print("predictions:", pred.ravel().tolist(), "targets:", y.ravel().astype(int).tolist())
print("accuracy:", float((pred == y).mean()))

# %% [markdown]
# The hidden layer has learned a representation in which XOR is linearly
# separable. The gradient check is not a formality: it certifies that the
# backward pass is the adjoint of the forward pass, which is the property every
# later interpretability claim depends on.
