import albumentations as A

from albumentations.pytorch import ToTensorV2

from video.utils.config import IMAGE_SIZE

train_transforms = A.Compose([

    A.RandomResizedCrop(
        size=(IMAGE_SIZE, IMAGE_SIZE),
        scale=(0.7, 1.0),
        ratio=(0.75, 1.33),
        p=1.0
    ),

    A.HorizontalFlip(p=0.5),

    A.Rotate(
        limit=20,
        p=0.5
    ),

    A.MotionBlur(
        blur_limit=7,
        p=0.3
    ),

    A.GaussNoise(
        std_range=(0.02, 0.08),
        p=0.3
    ),

    A.ImageCompression(
        quality_range=(40, 100),
        p=0.5
    ),

    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.4
    ),

    A.CoarseDropout(
        num_holes_range=(1, 5),
        hole_height_range=(20, 50),
        hole_width_range=(20, 50),
        fill=0,
        p=0.3
    ),

    A.Normalize(),

    ToTensorV2()
])

valid_transforms = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(),

    ToTensorV2()
])