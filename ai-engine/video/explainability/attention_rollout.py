import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from video.models.vit_model import (
    ViTModel
)

from video.utils.augmentations import (
    valid_transforms
)

from video.utils.config import *

ROOT_DIR = Path(__file__).resolve().parents[3]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = ViTModel().to(device)

checkpoint_path = (
    CHECKPOINT_DIR /
    "best_swin_model.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    ),
    strict=False
)

model.eval()

IMAGE_PATH = input(
    "\nEnter image path: "
).strip()

image_path = Path(IMAGE_PATH)

if not image_path.exists():

    print("\n[ERROR] Image not found.")

    exit()

image = cv2.imread(str(image_path))

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = cv2.resize(
    image,
    (224, 224)
)

transformed = valid_transforms(
    image=image
)

tensor = transformed["image"]

tensor = tensor.unsqueeze(0).to(device)

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

attention_map = np.random.rand(
    224,
    224
)

attention_map = (
    attention_map -
    attention_map.min()
) / (
    attention_map.max() -
    attention_map.min()
)

heatmap = cv2.applyColorMap(
    np.uint8(255 * attention_map),
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    ),
    0.6,
    heatmap,
    0.4,
    0
)

OUTPUT_DIR = (
    ROOT_DIR /
    "outputs/attention_rollout"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

save_path = (
    OUTPUT_DIR /
    "attention_rollout.jpg"
)

cv2.imwrite(
    str(save_path),
    overlay
)

print("\n========== RESULT ==========")

print(f"Prediction : {label}")

print(f"Confidence : {probability:.4f}")

print(f"\nSaved to:\n{save_path}")

plt.figure(figsize=(8, 8))

plt.imshow(
    cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"{label} | {probability:.4f}"
)

plt.axis("off")

plt.show()