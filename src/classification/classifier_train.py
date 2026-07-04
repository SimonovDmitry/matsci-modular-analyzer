import cv2
import albumentations as A
import random
import torch
import torchvision
import torchvision.datasets
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn
from sklearn.metrics import f1_score, classification_report
from pathlib import Path


class Classifier_train:
    def __init__(self, data_dir_train, data_dir_test):
        self.class_to_idx = {
            'otalkovannaya': 0,
            'ryadovaya': 1,
            'trudnoobogatimaya': 2
        }
        self.data_dir_train = Path(data_dir_train)
        self.data_dir_test = Path(data_dir_test)
        self.train_samples = []
        self.test_samples = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussNoise(
                std_range=(0.02, 0.08),
                mean_range=(0, 0),
                per_channel=True,
                p=0.4
            ),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(20, 60),
                hole_width_range=(20, 60),
                fill=0,
                p=0.5
            ),
        ])

    def fill_sets(self):
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir_train / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.*"):
                    self.train_samples.append((img_path, class_idx))
            else:
                print(f"Folder not found: {class_dir}")

            class_dir = self.data_dir_test / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.*"):
                    self.test_samples.append((img_path, class_idx))
            else:
                print(f"Folder not found: {class_dir}")

        random.shuffle(self.train_samples)
        random.shuffle(self.test_samples)

    def _preprocess_image(self, image, augment=False):
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        if augment:
            image = self.train_transform(image=image)['image']

        if image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def preprocess_train_image(self, image):
        return self._preprocess_image(image, augment=True)

    def preprocess_test_image(self, image):
        return self._preprocess_image(image, augment=False)

    def create_classifier(self):
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5, inplace=True),
            torch.nn.Linear(num_features, 3)
        )
        return model

    def _create_dataset(self, samples, augment=False):
        class CustomDataset(Dataset):
            def __init__(self, samples, preprocess_fn):
                self.samples = samples
                self.preprocess_fn = preprocess_fn

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                image = cv2.imread(str(img_path))
                if image is None:
                    raise ValueError(f"Failed to load: {img_path}")
                tensor = self.preprocess_fn(image)
                return tensor, label

        return CustomDataset(samples, self.preprocess_train_image if augment else self.preprocess_test_image)

    def get_train_loader(self, batch_size=16, shuffle=True):
        dataset = self._create_dataset(self.train_samples, augment=True)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, batch_size=16):
        dataset = self._create_dataset(self.test_samples, augment=False)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False)

    def get_metrics(self, data_loader, model, average='weighted'):
        tp = 0
        n = 0
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                n += labels.size(0)
                tp += (predicted == labels).sum()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        model.train()
        accuracy = tp / n
        f1 = f1_score(all_labels, all_preds, average=average)
        return {'accuracy': accuracy, 'f1_score': f1}

    def train_model(self, num_epochs, batch_size, lr, save_path):
        train_loader = self.get_train_loader(batch_size)
        test_loader = self.get_test_loader(batch_size)
        model = self.create_classifier().to(self.device)
        loss_function = torch.nn.CrossEntropyLoss()
        model.requires_grad_(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.05)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.requires_grad_().to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            metrics1 = self.get_metrics(train_loader, model)
            metrics2 = self.get_metrics(test_loader, model)
            print(f'Epoch[{epoch}]: train top-1 accuracy = {metrics1['accuracy']}')
            print(f'Test top-1 accuracy: {metrics2['accuracy']}')

            print(f'Epoch[{epoch}]: train f1 score = {metrics1['f1_score']}')
            print(f'Test f1 score : {metrics2['f1_score']}')

        torch.save(model.state_dict(), save_path)
        return model


def run_training_pipeline(data_dir_train, data_dir_test, num_epochs, batch_size, lr, save_path):
    classifier = Classifier_train(data_dir_train, data_dir_test)
    classifier.fill_sets()
    model = classifier.train_model(num_epochs, batch_size, lr, save_path)
    return model


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    data_dir_train = base_dir / 'data_train'
    data_dir_test = base_dir / 'data_test'

    model = run_training_pipeline(
        data_dir_train=data_dir_train,
        data_dir_test=data_dir_test,
        num_epochs=21,
        batch_size=16,
        lr=0.01,
        save_path='classifier4.pth'
    )
