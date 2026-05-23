import pandas as pd

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

datasets = [

    ROOT_DIR / "datasets/metadata/celebdf.csv",

    ROOT_DIR / "datasets/metadata/dfdc.csv",

    ROOT_DIR / "datasets/metadata/faceforensics.csv",

    ROOT_DIR / "datasets/metadata/deeperforensics.csv"
]

dfs = []

for dataset in datasets:

    if dataset.exists():

        df = pd.read_csv(dataset)

        dfs.append(df)

final_df = pd.concat(
    dfs,
    ignore_index=True
)

save_path = (
    ROOT_DIR /
    "datasets/metadata/final_dataset.csv"
)

final_df.to_csv(
    save_path,
    index=False
)

print(f"\nSaved merged dataset to:\n{save_path}")