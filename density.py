from transport import *
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Target density is un-normalized.
def p(x):
    x1 = x[0]
    x2 = x[1]
    V = 10*(x2 - x1**2)**2 + (x1-1)**2
    return np.exp(-V)

# Reference density is standard normal.
def b(x):
    x1 = x[0]
    x2 = x[1]
    V = 0.5*(x1**2 + x2**2) + np.log(2*np.pi)
    return np.exp(-V)

# Function to sample the reference distribution.
def sample_ref(N):
    return np.random.multivariate_normal(np.zeros(2), np.eye(2), N).T


N = 10000
X = sample_ref(N)

w0 = np.random.randn(9)#np.array([0., 10., 1., 1., 1., 10., 0., 1., 1.])

[w, f, g] = minimize_chisq(w0, X, p, b, 1e-3, 1000)

plt.figure(1)
plt.plot(f)

plt.figure(2)
plt.plot(g)

plt.show()
