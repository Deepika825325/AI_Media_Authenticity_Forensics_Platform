import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from video.models.swin_transformer import (
    SwinTransformerModel
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

print(f"\nUsing device: {device}")

model = SwinTransformerModel().to(device)

checkpoint_path = (
    CHECKPOINT_DIR /
    "best_swin_model.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.eval()

print("\nModel loaded successfully.")

target_layer = model.model.layers[-1]

cam = LayerCAM(
    model=model,
    target_layers=[target_layer]
)

IMAGE_PATH = input(
    "\nEnter image path: "
).strip()

image_path = Path(IMAGE_PATH)

if not image_path.exists():

    print("\n[ERROR] Image not found.")

    exit()

image = cv2.imread(str(image_path))

if image is None:

    print("\n[ERROR] Failed to load image.")

    exit()

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image_resized = cv2.resize(
    image_rgb,
    (224, 224)
)

rgb_float = (
    image_resized.astype(np.float32) / 255
)

transformed = valid_transforms(
    image=image_resized
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

confidence = (
    probability
    if probability > 0.5
    else 1 - probability
)

grayscale_cam = cam(
    input_tensor=tensor
)[0]

visualization = show_cam_on_image(
    rgb_float,
    grayscale_cam,
    use_rgb=True
)

OUTPUT_DIR = (
    ROOT_DIR /
    "outputs/layercam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

save_path = (
    OUTPUT_DIR /
    "layercam_result.jpg"
)

cv2.imwrite(
    str(save_path),
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)

print("\n========== RESULT ==========")

print(f"Prediction : {label}")

print(f"Confidence : {confidence:.4f}")

print(f"\nSaved to:\n{save_path}")

plt.figure(figsize=(8, 8))

plt.imshow(visualization)

plt.title(
    f"{label} | {confidence:.4f}"
)

plt.axis("off")

plt.show()