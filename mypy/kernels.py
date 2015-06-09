from __future__ import division

import numpy as np

from util import as_row, as_col


class Kernel:
    def __init__(self, kern_func, **param):
        self.kern_func = kern_func
        self.param = dict(param)

    def eval(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        d = pairwise_differences(x1, x2):
        return self.kern_func(d, **self.param)


class AdditiveKernel:
    def __init__(self, *kernels):
        self.kernels = kernels

    def eval(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        d = pairwise_differences(x1, x2):
        matrices = [k(d) for k in self.kernels]
        return sum(matrices)


def noise(d, variance=1.0):
    m = np.zeros_like(d)
    m[d == 0.0] = variance
    return m


def constant(d, variance):
    return variance * np.ones_like(d)


def constant(d, variance):
    return {'variance': np.ones_like(d)}


def squared_exponential(d, variance, length):
    r = np.abs(d)
    return variance * np.exp(-0.5 * (r / length) ** 2)


def grad_squared_exponential(d, variance, length):
    r = np.abs(d)
    g = dict()
    g['variance'] = squared_exponential(d, 1.0, length)
    g['length']   = squared_exponential(d, variance, length)
    g['length']  *= (r ** 2) / (length ** 3)
    return g


def exponential(d, variance, length):
    r = np.abs(d)
    return variance * np.exp(- r / length)


def grad_exponential(d, variance, length):
    r = np.abs(d)
    g = dict()
    g['variance'] = exponential(d, 1.0, length)
    g['length']   = exponential(d, variance, length)
    g['length']  *= r / (length ** 2)
    return g


def pairwise_differences(x1, x2):
    """Compute differences between all pairs of elements in two vectors.

    # Arguments

    x1 : A sequence of numbers.
    x2 : A sequence of numbers.

    # Returns

    A matrix with `len(x1)` rows and `len(x2)` columns.

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return as_col(x1) - as_row(x2)
