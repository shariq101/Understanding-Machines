# %% [markdown]
# # Lab 1 — The line and its limits
#
# A perceptron learns a half-space. On linearly separable data the update
# converges; on XOR it cannot. This lab implements the update, trains it on
# three Boolean functions, and makes the failure visible.
#
# **Milestone:** explain why XOR sits at exactly chance while AND and OR reach 1.0.

# %%
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = {
    "AND": np.array([-1, -1, -1, 1], dtype=float),
    "OR":  np.array([-1, 1, 1, 1], dtype=float),
    "XOR": np.array([-1, 1, 1, -1], dtype=float),
}


def perceptron(X, y, epochs=20, lr=0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:
                w += lr * yi * xi
                b += lr * yi
    return w, b


def accuracy(X, y, w, b):
    pred = np.where(X @ w + b >= 0, 1, -1)
    return float((pred == y).mean())


for name, y in Y.items():
    w, b = perceptron(X, y)
    print(f"{name:3s}  w={np.round(w, 2)}  b={b:+.1f}  accuracy={accuracy(X, y, w, b):.2f}")

# %% [markdown]
# The AND and OR boundaries separate the square. XOR cannot: its two positive
# corners are opposite each other, and a half-space is convex, so any plane that
# contains both must also contain the midpoint (0.5, 0.5), which is negative.
#
# The update is the gradient of the hinge loss `max(0, -y(w·x+b))`, so this is
# not a different algorithm from the one in the chapter; it is the same loss
# with a step-size schedule.

# %%
print("midpoint label in XOR:", Y["XOR"][np.argmin(np.abs(X - 0.5).sum(1))])
print("half-space is convex: a positive region containing (0,1) and (1,0)")
print("must contain their midpoint (0.5, 0.5), whose XOR label is -1")
