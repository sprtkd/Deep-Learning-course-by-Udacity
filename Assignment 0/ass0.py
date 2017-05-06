"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    exparr = np.exp(x)
    f = exparr/exparr.sum(axis=0)
    return f


#print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print(softmax(scores/10))
print(sum(softmax(scores)))
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
