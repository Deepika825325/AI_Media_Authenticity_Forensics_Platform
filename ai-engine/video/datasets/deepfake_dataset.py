import cv2
import pandas as pd

from torch.utils.data import Dataset

class DeepFakeDataset(Dataset):

    def __init__(self, csv_path, transforms=None):

        self.df = pd.read_csv(csv_path)

        self.transforms = transforms

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = cv2.imread(row["path"])

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label = row["label"]

        if self.transforms:

            transformed = self.transforms(
                image=image
            )

            image = transformed["image"]

        return image, label