"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math

def softMaxHelper(x):
    return map(lambda y: math.exp(y)/(math.fsum(map(math.exp , x))), x)

def softmax(x):
    x = np.array(x)
    if (x.ndim == 1):
        return softMaxHelper(x)
    elif (x.ndim == 2):
        outlist = list()
        for y in range (0, x.T.shape[0]):
            outlist.append(softMaxHelper(x.T[y]))
        return np.vstack(outlist).T
    return x

print(softmax(scores))


# Plot softmax curves

import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
