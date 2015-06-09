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

