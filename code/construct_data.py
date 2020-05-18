"""
Usage
-----

python construct_data.py
python construct_data.py --test
python construct_data.py --combine

"""

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

NUM_FEATURES = 5000
NUM_CLASSES = 3993

# Some rows don't have any labels
train_skiprows = [
    94,
    253,
    510,
    1528,
    1940,
    1954,
    4030,
    4427,
    4644,
    4728,
    5232,
    5763,
    6296,
    6334,
    6704,
    7084,
    9478,
    9676,
    10000,
    10489,
    10737,
    10911,
    11675,
    12281,
    12600,
    13148,
    14168,
    14723,
]

dev_skiprows = [194, 415]


def construct(path, train=True):
    """TODO"""
    file_prefix = "train"
    skiprows = train_skiprows
    if not train:
        file_prefix = "dev"
        skiprows = dev_skiprows

    # Default path for saving the dataframes
    Path("../data/expanded").mkdir(parents=True, exist_ok=True)

    # We don't need the column "ex_id"
    raw_data = pd.read_csv(
        f"{path}/{file_prefix}.csv", usecols=["labels", "features"], skiprows=skiprows,
    )

    # Formatting labels
    raw_data["labels"] = (
        raw_data["labels"].str.split(",").apply(lambda x: list(map(int, x)))
    )

    # Formatting features
    raw_data["features"] = (
        raw_data["features"]
        .str.split(" ")
        .apply(lambda x: {int(i.split(":")[0]): float(i.split(":")[1]) for i in x})
    )

    raw_data["features"] = raw_data["features"].apply(
        lambda x: [list(x.keys()), list(x.values())]
    )

    temp = np.zeros((raw_data.shape[0], NUM_FEATURES))
    features = pd.DataFrame(temp)

    temp = np.zeros((raw_data.shape[0], NUM_CLASSES))
    labels = pd.DataFrame(temp)

    def create_features(row):
        features.loc[row.name, row["features"][0]] = row["features"][1]

    def create_labels(row):
        labels.loc[row.name, row["labels"]] = 1

    # Creating features and labels dataframes
    print("Creating features...")
    raw_data.progress_apply(lambda x: create_features(x), axis=1)
    print("Creating labels...")
    raw_data.progress_apply(lambda x: create_labels(x), axis=1)

    # Saving training/dev dataframes
    features.to_csv(
        f"../data/expanded/{file_prefix}_features.csv", index=False, header=False
    )
    labels.to_csv(
        f"../data/expanded/{file_prefix}_labels.csv", index=False, header=False
    )
    print(f"Saved dataframes.")


def construct_test_data(path):
    """TODO"""
    # Default path for saving the dataframes
    Path("../data/expanded").mkdir(parents=True, exist_ok=True)

    # We don't need the column "ex_id" and "labels"
    raw_data = pd.read_csv(f"{path}/test_no_label.csv", usecols=["features"])

    # Formatting features
    raw_data["features"] = (
        raw_data["features"]
        .str.split(" ")
        .apply(lambda x: {int(i.split(":")[0]): float(i.split(":")[1]) for i in x})
    )

    raw_data["features"] = raw_data["features"].apply(
        lambda x: [list(x.keys()), list(x.values())]
    )

    temp = np.zeros((raw_data.shape[0], NUM_FEATURES))
    features = pd.DataFrame(temp)

    def create_features(row):
        features.loc[row.name, row["features"][0]] = row["features"][1]

    # Creating features dataframe
    print("Creating features...")
    raw_data.progress_apply(lambda x: create_features(x), axis=1)

    # Saving testing dataframe
    features.to_csv(f"../data/expanded/test_features.csv", index=False, header=False)
    print(f"Saved dataframe.")


def combine_datasets():
    X_train = pd.read_csv(
        "../data/expanded/train_features.csv", names=range(NUM_FEATURES)
    )
    y_train = pd.read_csv("../data/expanded/train_labels.csv", names=range(NUM_CLASSES))
    X_valid = pd.read_csv(
        "../data/expanded/dev_features.csv", names=range(NUM_FEATURES)
    )
    y_valid = pd.read_csv("../data/expanded/dev_labels.csv", names=range(NUM_CLASSES))

    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = pd.concat([y_train, y_valid], ignore_index=True)

    X_combined.to_csv(
        "../data/expanded/combined_features.csv", index=False, header=False
    )
    y_combined.to_csv("../data/expanded/combined_labels.csv", index=False, header=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--combine", action="store_true", default=False)
    args = parser.parse_args()

    if args.combine:
        print("Combining training and validation datasets: ")
        combine_datasets()
    else:
        if not args.test:
            print("Constructing training data:")
            construct("../data/raw")
            print("Constructing development data:")
            construct("../data/raw", train=False)
        else:
            print("Constructing testing data:")
            construct_test_data("../data/raw")
