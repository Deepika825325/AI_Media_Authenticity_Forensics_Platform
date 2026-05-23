import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
    transforms=train_transforms
)

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = (
    torch.utils.data.random_split(
        dataset,
        [train_size, valid_size]
    )
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================
# MODEL
# =========================================

model = DeepFakeModel().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================================
# TRAIN LOOP
# =========================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(device)

        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    print(
        f"\nEpoch {epoch+1} Loss: "
        f"{running_loss/len(train_loader)}"
    )

# =========================================
# SAVE MODEL
# =========================================

torch.save(
    model.state_dict(),
    CHECKPOINT_DIR / "deepfake_detector.pth"
)

print("\nModel saved successfully.")