import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


class AugmentationPipeline:
    def __init__(self, image_size = 512):
        self.image_size = image_size

    def get_train_pipeline(self):
        return A.Compose([
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.5, 1.0),
                p=1.0
            ),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),

            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomShadow(p=0.3),

            A.OneOf([
                A.CoarseDropout(
                    num_holes_range=(4, 12),
                    hole_height_range=(10, 40),
                    hole_width_range=(10, 40),
                    fill_value=0,
                    p=1.0
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.4),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def get_val_pipeline(self):
        return A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def apply(self, image, mask, is_train = True):
        pipeline = self.get_train_pipeline() if is_train else self.get_val_pipeline()
        augmented = pipeline(image=image, mask=mask)
        return augmented["image"], augmented["mask"]