"""
Usage
-----

python train.py --units 256 --dropout 0.2 --epochs 25

"""

import random
from argparse import ArgumentParser

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

        self.classifier = nn.Sequential(
            nn.Linear(NUM_FEATURES, units),
            # nn.BatchNorm1d(units),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            nn.Linear(units, NUM_CLASSES),
        )

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
    trainer = Trainer(gpus=1, max_epochs=args.epochs, auto_lr_find=True)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    # parametrize the network
    parser.add_argument("--units", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2_penalty", type=float, default=0.0)

    args = parser.parse_args()

    # Train
    main(args)
