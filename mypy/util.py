import numpy as np


def as_row(x):
    return x.ravel()[np.newaxis, :]


def as_col(x):
    return as_row(x).T


def days_between(now, then):
    try:
        return (now - then).days
    except AttributeError:
        return np.nan


def check_grad(f, x0, eps=1e-8):
    f0 = f(x0)
    n = x0.size
    g = np.zeros_like(x0)
    for i in range(n):
        dt = np.zeros_like(x0)
        dt[i] += eps
        f1 = f(x0 + dt)
        g[i] = (f1 - f0) / eps
        
    return g

