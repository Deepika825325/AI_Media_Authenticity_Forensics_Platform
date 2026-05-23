import pandas as pd

from pathlib import Path

DATASET_DIR = Path(
    "/content/drive/MyDrive/DeepVerify/datasets"
)

csv_path = (
    DATASET_DIR /
    "processed/metadata/dataset.csv"
)

df = pd.read_csv(csv_path)

def fix_path(old_path):

    old_path = old_path.replace("\\", "/")

    if "aligned_faces/" in old_path:

        relative = old_path.split(
            "aligned_faces/"
        )[1]

        return str(
            DATASET_DIR /
            "processed/aligned_faces" /
            relative
        )

    return old_path

df["image_path"] = df["image_path"].apply(
    fix_path
)

save_path = (
    DATASET_DIR /
    "processed/metadata/dataset_colab.csv"
)

df.to_csv(
    save_path,
    index=False
)

print("\nSaved fixed CSV to:")

print(save_path)