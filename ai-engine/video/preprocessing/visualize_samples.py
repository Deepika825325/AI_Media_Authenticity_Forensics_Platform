import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# =========================================
# ROOT DIRECTORY
# =========================================

ROOT_DIR = Path(__file__).resolve().parents[3]

# =========================================
# DATASET PATHS
# =========================================

DATA_DIR = ROOT_DIR / "datasets/processed/aligned_faces"

# =========================================
# CONFIG
# =========================================

NUM_SAMPLES = 6

# =========================================
# LOAD RANDOM IMAGES
# =========================================

images = []

for label in ["real", "fake"]:

    label_dir = DATA_DIR / label

    if not label_dir.exists():
        print(f"[ERROR] Missing directory: {label_dir}")
        continue

    all_images = list(label_dir.glob("*/*.jpg"))

    selected = random.sample(
        all_images,
        min(NUM_SAMPLES, len(all_images))
    )

    for image_path in selected:

        images.append({
            "path": image_path,
            "label": label
        })

# =========================================
# VISUALIZATION
# =========================================

plt.figure(figsize=(15, 8))

for idx, item in enumerate(images):

    image = cv2.imread(str(item["path"]))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2, NUM_SAMPLES, idx + 1)

    plt.imshow(image)

    plt.title(item["label"].upper())

    plt.axis("off")

plt.tight_layout()

# =========================================
# SAVE OUTPUT
# =========================================

OUTPUT_DIR = ROOT_DIR / "datasets/processed/visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

save_path = OUTPUT_DIR / "sample_visualization.png"

plt.savefig(save_path)

print(f"\nVisualization saved to:\n{save_path}")

plt.show()