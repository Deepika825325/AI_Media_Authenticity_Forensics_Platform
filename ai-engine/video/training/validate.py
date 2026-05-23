import torch
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import DataLoader

from video.datasets.deepfake_dataset import (
    DeepFakeDataset
)

from video.models.xceptionnet import (
    DeepFakeModel
)

from video.utils.config import *
from video.utils.augmentations import *

# =========================================
# DEVICE
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# =========================================
# DATASET
# =========================================

dataset = DeepFakeDataset(
    CSV_PATH,
    transforms=valid_transforms
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================
# MODEL
# =========================================

model = DeepFakeModel().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT_DIR / "deepfake_detector.pth",
        map_location=device
    )
)

model.eval()

# =========================================
# VALIDATION
# =========================================

y_true = []
y_pred = []

with torch.no_grad():

    loop = tqdm(loader)

    for images, labels in loop:

        images = images.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)

        preds = (preds > 0.5).int()

        y_true.extend(labels.numpy())

        y_pred.extend(
            preds.cpu().numpy().flatten()
        )

# =========================================
# METRICS
# =========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

print("\n========== RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")

print(confusion_matrix(y_true, y_pred))