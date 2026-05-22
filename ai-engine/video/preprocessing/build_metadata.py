import pandas as pd
from pathlib import Path

# =========================================
# ROOT DIRECTORY
# =========================================

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "datasets/processed/aligned_faces"

OUTPUT_CSV = (
    ROOT_DIR /
    "datasets/processed/metadata/dataset.csv"
)

rows = []

# =========================================
# BUILD METADATA
# =========================================

for label in ["real", "fake"]:

    label_value = 0 if label == "real" else 1

    label_dir = DATA_DIR / label

    if not label_dir.exists():

        print(f"[ERROR] Missing directory: {label_dir}")

        continue

    for video_folder in label_dir.iterdir():

        if not video_folder.is_dir():
            continue

        for image_path in video_folder.glob("*.jpg"):

            rows.append({
                "path": str(image_path),
                "label": label_value,
                "video": video_folder.name
            })

# =========================================
# SAVE CSV
# =========================================

df = pd.DataFrame(rows)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nMetadata CSV generated successfully.")
print(df.head())

print(f"\nSaved CSV to:\n{OUTPUT_CSV}")