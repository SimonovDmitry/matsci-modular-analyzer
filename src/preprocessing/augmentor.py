import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class AugmentationPipeline:
    def __init__(self, image_size = 512):
        self.image_size = image_size

    def get_train_pipeline(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.1, 0.1),
                rotate=(-180, 180),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),

            A.OneOf([
                A.CoarseDropout(
                    num_holes_range=(1, 12),
                    hole_height_range=(1, 3),
                    hole_width_range=(10, 150),
                    fill=0
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(1, 2),
                    hole_width_range=(5, 80),
                    fill=200
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 12),
                    hole_height_range=(2, 10),
                    hole_width_range=(2, 10),
                    fill=0
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(10, 50),
                    hole_width_range=(10, 50),
                    fill=0
                )
            ], p=0.6),

            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 0.7),
                p=0.2
            ),

            A.ElasticTransform(
                alpha=5,
                sigma=20,
                p=0.15
            ),

            A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2)),
                A.GaussNoise(
                    std_range=(0.05, 0.15),
                    mean_range=(0, 0),
                    per_channel=True,
                    p=0.4
                )
            ], p=0.4),

            A.GaussianBlur(blur_limit=(3, 5), p=0.2)
        ], mask_interpolation=cv2.INTER_NEAREST)

    def augment_generate(self, image, mask, num_augmentations_per_image=10, output_size=(512, 512)):
        image = cv2.resize(image, output_size)
        mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

        transform = self.get_train_pipeline()
        images = [image]
        masks = [mask]

        for _ in range(num_augmentations_per_image):
            augmented = transform(image=image, mask=mask)
            images.append(augmented['image'])
            masks.append(augmented['mask'])

        images = np.array(images)
        masks = np.array(masks)

        return images, masks

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
