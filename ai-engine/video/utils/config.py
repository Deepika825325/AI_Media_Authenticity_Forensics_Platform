from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

# =========================================
# DATA
# =========================================

CSV_PATH = (
    ROOT_DIR /
    "datasets/metadata/final_dataset.csv"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

# =========================================
# TRAINING
# =========================================

EPOCHS = 10
LEARNING_RATE = 1e-4

# =========================================
# PATHS
# =========================================

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)