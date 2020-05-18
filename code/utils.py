import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


def LRAP(y_true, y_score):
    """TODO"""
    return label_ranking_average_precision_score(y_true, y_score)


def perfection(y_true, y_pred):
    if type(y_true) is not np.ndarray:
        y_true = y_true.to_numpy()
    if type(y_pred) is not np.ndarray:
        y_pred = y_pred.to_numpy()
    assert len(y_true) == len(y_pred)

    num_samples = len(y_true)
    perfect = 0
    for i in range(num_samples):
        if np.alltrue(y_true[i] == y_pred[i]):
            perfect += 1

    return perfect / num_samples
