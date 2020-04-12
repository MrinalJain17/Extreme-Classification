import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import train_test_split


def split(features, labels, test_size=0.1, random_state=42):
    """TODO"""
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    train_features.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)
    valid_features.reset_index(drop=True, inplace=True)
    valid_labels.reset_index(drop=True, inplace=True)

    return train_features, valid_features, train_labels, valid_labels


def LRAP(y_true, y_score):
    """TODO"""
    return label_ranking_average_precision_score(y_true, y_score)
