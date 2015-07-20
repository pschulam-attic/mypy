'An implementation of softmax regression.'

import numpy as np

from scipy.misc import logsumexp

from ..util import as_row, as_col


def regression_proba(x, W):
    z = np.dot(W, x)
    p = softmax_func(z)
    return p


def regression_log_proba(x, W):
    p = regression_proba(x, W)
    return np.log(p)


def regression_ll(x, y, W):
    p = regression_proba(x, W)
    return sum(y * np.log(p))


def regression_ll_grad(x, y, W):
    z = np.dot(W, x)
    p = softmax_func(z)
    G = softmax_grad(z)
    D = np.zeros_like(W)

    for i, _ in enumerate(D):
        if i == 0: continue
        for j, _ in enumerate(p):
            D[i] += y[j] / p[j] * G[i, j] * x

    return D


def softmax_func(z):
    p = np.exp(z - logsumexp(z))
    return p


def softmax_grad(z):
    p = softmax_func(z)
    g = np.diag(p) - as_col(p) * as_row(p)
    return g


def onehot_encode(y, k):
    e = np.zeros(k)
    e[y] = 1
    return e
