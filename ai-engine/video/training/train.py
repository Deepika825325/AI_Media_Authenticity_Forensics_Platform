import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from torch.utils.data import (
    DataLoader,
    random_split
)

from video.datasets.deepfake_dataset import (
    DeepFakeDataset
)

from video.models.swin_transformer import (
    SwinTransformerModel
)

from video.utils.config import *
from video.utils.augmentations import *

ROOT_DIR = Path(__file__).resolve().parents[3]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")

dataset = DeepFakeDataset(
    CSV_PATH,
    transforms=train_transforms
)

train_size = int(0.8 * len(dataset))

valid_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(
    dataset,
    [train_size, valid_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

model = SwinTransformerModel().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(device)

        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

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

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    scheduler.step()

    print(
        f"\nEpoch {epoch+1} Loss: "
        f"{epoch_loss:.6f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        save_path = (
            CHECKPOINT_DIR /
            "best_swin_model.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )

        print(
            f"\nBest model saved to:\n{save_path}"
        )

print("\nTraining completed.")