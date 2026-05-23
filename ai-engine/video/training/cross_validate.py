import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit
)

from video.utils.config import *

df = pd.read_csv(CSV_PATH)

splitter = GroupShuffleSplit(
    test_size=0.2,
    n_splits=1
)

groups = df["dataset"]

train_idx, valid_idx = next(

    splitter.split(
        df,
        groups=groups
    )
)

train_df = df.iloc[train_idx]

valid_df = df.iloc[valid_idx]

print("\nTrain datasets:")

print(train_df["dataset"].value_counts())

print("\nValidation datasets:")

print(valid_df["dataset"].value_counts())