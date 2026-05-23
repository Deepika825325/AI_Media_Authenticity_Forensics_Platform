from pathlib import Path

ROOT_DIR = Path("/content/AI_Media_Authenticity_Forensics_Platform")

DATASET_DIR = Path(
    "/content/drive/MyDrive/DeepVerify/datasets"
)

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/DeepVerify/checkpoints"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/DeepVerify/outputs"
)

LOG_DIR = Path(
    "/content/drive/MyDrive/DeepVerify/logs"
)

CSV_PATH = (
    DATASET_DIR /
    "metadata/final_dataset.csv"
)

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 1e-4