"""
Usage
-----

python train.py --units 512 256 --dropout 0.5 0.2 --epochs 50
python train.py --units 256 --dropout 0.2 --epochs 50 --lr 0.01

"""

import random
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer

from dataset import CSVDataset
from utils import LRAP

SEED = 42

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

NUM_FEATURES = 5000
NUM_CLASSES = 3993


class Net(pl.LightningModule):
    """TODO

    """

    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams

        units = self.hparams.units
        dropout = self.hparams.dropout

        # Supports only 1 or 2-layer architectures for now
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("dropout_input", nn.Dropout(0.15)),
                    ("fc1", nn.Linear(NUM_FEATURES, units[0])),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout[0])),
                ]
            )
        )
        if len(units) > 1:
            self.classifier.add_module("fc2", nn.Linear(units[0], units[-1]))
            self.classifier.add_module("relu2", nn.ReLU())
            self.classifier.add_module("dropout2", nn.Dropout(dropout[-1]))

        self.classifier.add_module("output", nn.Linear(units[-1], NUM_CLASSES))

    def forward(self, x):
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.forward(features)
        loss = F.multilabel_soft_margin_loss(predictions, labels)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.forward(features)
        loss = F.multilabel_soft_margin_loss(predictions, labels)

        with torch.no_grad():
            pred = torch.sigmoid(predictions).cpu().numpy()
            actual = labels.cpu().numpy()

            lrap_score = torch.tensor(LRAP(actual, pred))

        return {"val_loss": loss, "LRAP": lrap_score}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        lrap_mean = torch.stack([x["LRAP"] for x in outputs]).mean()

        logs = {"val_loss": val_loss_mean, "LRAP": lrap_mean}
        return {"val_loss": val_loss_mean, "log": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_penalty
        )

    def prepare_data(self):
        self.train_data = CSVDataset(
            "../data/expanded/", standardize="./saved_models/scaler.pkl"
        )
        self.validation_data = CSVDataset(
            "../data/expanded/",
            csv_features="dev_features.csv",
            csv_labels="dev_labels.csv",
            standardize="./saved_models/scaler.pkl",
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_data, batch_size=self.hparams.batch_size
        )


def main(args):
    model = Net(args)
    trainer = Trainer(gpus=1, max_epochs=args.epochs)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument("--units", type=int, nargs="+", default=512)
    parser.add_argument(
        "--dropout", type=float, nargs="+", default=0.4
    )  # Should be specified for each fully-connected layer
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2_penalty", type=float, default=0.0)

    args = parser.parse_args()

    if not isinstance(args.units, list):
        args.units = [args.units]
    if not isinstance(args.dropout, list):
        args.dropout = [args.dropout]
    assert len(args.units) == len(args.dropout)

    # Train
    main(args)
