import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from video.models.xceptionnet import (
    DeepFakeModel
)

from video.utils.augmentations import (
    valid_transforms
)

from video.utils.config import *

# =========================================
# ROOT DIRECTORY
# =========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

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
# TARGET LAYER
# =========================================

target_layer = model.model.conv_head

# =========================================
# INITIALIZE GRADCAM
# =========================================

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

# =========================================
# INPUT IMAGE
# =========================================

IMAGE_PATH = input(
    "\nEnter image path: "
).strip()

image_path = Path(IMAGE_PATH)

if not image_path.exists():

    print("\n[ERROR] Image path not found.")

    exit()

# =========================================
# LOAD IMAGE
# =========================================

image = cv2.imread(str(image_path))

if image is None:

    print("\n[ERROR] Failed to load image.")

    exit()

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# =========================================
# RESIZE FOR OVERLAY
# =========================================

original_resized = cv2.resize(
    image_rgb,
    (224, 224)
)

rgb_float = (
    original_resized.astype(np.float32) / 255
)

# =========================================
# PREPROCESS
# =========================================

transformed = valid_transforms(
    image=original_resized
)

tensor = transformed["image"]

tensor = tensor.unsqueeze(0).to(device)

# =========================================
# MODEL PREDICTION
# =========================================

with torch.no_grad():

    output = model(tensor)

    probability = torch.sigmoid(
        output
    ).item()

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

print("\n========== RESULT ==========")

print(f"Prediction : {label}")

print(f"Confidence : {confidence:.4f}")

print("============================")

# =========================================
# GENERATE GRADCAM
# =========================================

grayscale_cam = cam(
    input_tensor=tensor
)[0]

# =========================================
# OVERLAY HEATMAP
# =========================================

visualization = show_cam_on_image(
    rgb_float,
    grayscale_cam,
    use_rgb=True
)

# =========================================
# SAVE OUTPUT
# =========================================

OUTPUT_DIR = ROOT_DIR / "outputs/heatmaps"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

save_path = (
    OUTPUT_DIR /
    "gradcam_result.jpg"
)

cv2.imwrite(
    str(save_path),
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print(f"\nHeatmap saved to:\n{save_path}")

# =========================================
# DISPLAY RESULT
# =========================================

plt.figure(figsize=(8, 8))

plt.imshow(visualization)

plt.title(
    f"{label} | Confidence: {confidence:.4f}"
)

plt.axis("off")

plt.show()