import cv2
import torch

from pathlib import Path

from video.models.xceptionnet import (
    DeepFakeModel
)

from video.utils.augmentations import (
    valid_transforms
)

from video.utils.config import *

# =========================================
# DEVICE
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")

# =========================================
# LOAD MODEL
# =========================================

model = DeepFakeModel().to(device)

checkpoint_path = (
    CHECKPOINT_DIR /
    "deepfake_detector.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.eval()

print("\nModel loaded successfully.")

# =========================================
# IMAGE PATH
# =========================================

IMAGE_PATH = input(
    "\nEnter image path: "
).strip()

# =========================================
# CHECK IMAGE EXISTS
# =========================================

image_path = Path(IMAGE_PATH)

if not image_path.exists():

    print("\n[ERROR] Image path does not exist.")

    print(f"\nProvided path:\n{image_path}")

    exit()

# =========================================
# LOAD IMAGE
# =========================================

image = cv2.imread(str(image_path))

if image is None:

    print("\n[ERROR] Failed to load image.")

    exit()

# =========================================
# PREPROCESS
# =========================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

transformed = valid_transforms(
    image=image_rgb
)

tensor = transformed["image"]

tensor = tensor.unsqueeze(0).to(device)

# =========================================
# PREDICTION
# =========================================

with torch.no_grad():

    output = model(tensor)

    probability = torch.sigmoid(
        output
    ).item()

# =========================================
# LABEL
# =========================================

label = (
    "FAKE"
    if probability > 0.5
    else "REAL"
)

confidence = (
    probability
    if probability > 0.5
    else 1 - probability
)

# =========================================
# OUTPUT
# =========================================

print("\n========== RESULT ==========")

print(f"Prediction : {label}")

print(f"Confidence : {confidence:.4f}")

print(f"Raw Score  : {probability:.4f}")

print("============================")