from __future__ import division

import numpy as np
import pandas as pd
from sklearn import metrics

class Score():
    '''
    Only provide binary input
    Scores are calculated based on 1
    '''
    
    @staticmethod
    def recall(y_true, y_pred, reverse=False):
        ''' tp / (tp+fn) --> Accuracy(y_true = 1 | y_pred = 1) '''
        if reverse:
            y_true = 1 - y_true
            y_pred = 1 - y_pred
        return metrics.recall_score(y_true, y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, reverse=False):
        ''' tp / (tp+fp) '''
        if reverse:
            y_true = 1 - y_true
            y_pred = 1 - y_pred
        return metrics.precision_score(y_true, y_pred)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    
    @staticmethod
    def auc(y_true, y_pred):
        return metrics.roc_auc_score(y_true, y_pred)
    
    @staticmethod
    def roc_curve(y_true, y_pred):
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        return fpr, tpr, threshold


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