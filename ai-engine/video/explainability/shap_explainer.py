import cv2
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

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

background = tensor[:1]

explainer = shap.DeepExplainer(
    model,
    background
)

shap_values = explainer.shap_values(
    tensor
)

shap_image = np.transpose(
    tensor[0].cpu().numpy(),
    (1, 2, 0)
)

plt.figure(figsize=(8, 8))

shap.image_plot(
    shap_values,
    np.expand_dims(shap_image, axis=0),
    show=False
)

OUTPUT_DIR = (
    ROOT_DIR /
    "outputs/shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

save_path = (
    OUTPUT_DIR /
    "shap_result.png"
)

plt.savefig(save_path)

print(f"\nSaved SHAP result to:\n{save_path}")

plt.show()