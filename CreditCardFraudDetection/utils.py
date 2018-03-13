from __future__ import division

import numpy as np
import pandas as pd

def to_one_hot(y):
    if isinstance(y, np.ndarray):
        y = y.flatten()
    y = np.array(y, dtype=int)
    n_values = int(np.max(y)) + 1
    y = np.eye(n_values)[y]
    return y


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    np.random.seed(172)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def standarize():
    pass