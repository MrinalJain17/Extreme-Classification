# Extreme-Classification

DS-GA 1003 (Machine Learning) - Group project on "Extreme Classification"

## Dataset

1. Download `train.csv` and `dev.csv` from [here](https://worksheets.codalab.org/worksheets/0x0a35e4ca487b4892976188108704011c).
2. Place the files inside the directory `data/raw/` in the root of the repository. (Recommended)
3. Run the following command to convert sparse data to "normal" dataframes, from the root of the repository.

   ```bash
   cd code
   python construct_data.py
   ```
4. The dataframes can then be loaded as follows:

   ```python
   import pandas as pd
   
   # Assuming we are in one of the sub-directories (code, notebooks, etc)
   features = pd.read_csv("../data/expanded/train_features.csv")
   labels = pd.read_csv("../data/expanded/train_labels.csv")
   ```

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm
