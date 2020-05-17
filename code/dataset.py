from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_FEATURES = 5000
NUM_CLASSES = 3993

cwd = Path.cwd()


class CSVDataset(Dataset):
    def __init__(
        self,
        root_dir,
        csv_features="train_features.csv",
        csv_labels="train_labels.csv",
        standardize=None,
    ):
        self.root_dir = Path(root_dir)
        self.features = (
            pd.read_csv(self.root_dir / csv_features, names=range(NUM_FEATURES))
            .to_numpy()
            .astype("float32")
        )
        self.labels = (
            pd.read_csv(self.root_dir / csv_labels, names=range(NUM_CLASSES))
            .to_numpy()
            .astype("float32")
        )

        self.features = np.log1p(self.features)

        if standardize is not None:
            scaler = load(open(standardize, "rb"))
            self.features = scaler.transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_features = torch.from_numpy(self.features[idx, :])
        batch_labels = torch.from_numpy(self.labels[idx, :])

        return batch_features, batch_labels


# Used this piece of code to obtain the class weights, as required for BCE loss in PyTorch
#
# y_train = pd.read_csv("../../data/expanded/train_labels.csv", names=range(NUM_CLASSES))
#
# num_samples = y_train.shape[0]
# class_weight = (num_samples - y_train.sum()) / y_train.sum()
# class_weight = class_weight.to_numpy().astype("float32")
# np.savez("class_weight", class_weight=class_weight)
