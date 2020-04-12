import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


def split(features, labels, test_split=0.2, random_state=42):
    """TODO"""
    rng = np.random.default_rng(random_state)
    num_examples = features.shape[0]
    num_valid = int(num_examples * test_split)

    # Randomly dividing between training and validation data
    index = np.arange(num_examples)
    rng.shuffle(index)
    valid_index = index[:num_valid]
    train_index = index[num_valid:]

    train_features = features.loc[train_index].reset_index(drop=True)
    train_labels = labels.loc[train_index].reset_index(drop=True)
    valid_features = features.loc[valid_index].reset_index(drop=True)
    valid_labels = labels.loc[valid_index].reset_index(drop=True)

    return train_features, valid_features, train_labels, valid_labels


def LRAP(y_true, y_score):
    """TODO"""
    return label_ranking_average_precision_score(y_true, y_score)
