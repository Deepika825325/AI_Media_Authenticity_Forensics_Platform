import albumentations as A
from albumentations.pytorch import ToTensorV2

from video.utils.config import IMAGE_SIZE

train_transforms = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.RandomBrightnessContrast(p=0.3),

    A.GaussianBlur(p=0.2),

    A.ImageCompression(
        quality_range=(60, 100),
        p=0.3
    ),

    A.Normalize(),

    ToTensorV2()
])

valid_transforms = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(),

    ToTensorV2()
])