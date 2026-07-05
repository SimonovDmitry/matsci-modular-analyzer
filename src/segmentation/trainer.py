import os
import torch
import torch.optim as optim
import logging
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .model import TalcSegmentationModel
from .dataset import TalcDataset
from .losses import TalcLoss
from .metrics import TalcMetrics
from src.preprocessing import AugmentationPipeline


class SegmentationTrainer:
    def __init__(self, model, optimizer, criterion, device, metrics_helper, logger):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics_helper
        self.logger = logger

    def run_epoch(self, dataloader, epoch, is_train=True):
        self.model.train() if is_train else self.model.eval()
        self.metrics.reset()
        total_loss = 0
        loader = tqdm(dataloader, desc=f"Epoch {epoch} [{'Train' if is_train else 'Val'}]")
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            with torch.set_grad_enabled(is_train):
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)
                if is_train:
                    self.optimizer.zero_grad();
                    loss.backward();
                    self.optimizer.step()
            total_loss += loss.item()
            self.metrics.update(logits, masks)
            loader.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(dataloader), self.metrics.compute_epoch_metrics()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training")

def train_pipeline(data_root, warmup=15, fine_tune=35, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = TalcDataset.from_root(data_root, logger=logger)
    indices = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    aug = AugmentationPipeline(image_size=512)
    train_ds = TalcDataset([full_ds.image_paths[i] for i in train_idx], [full_ds.mask_paths[i] for i in train_idx],
                           transform=aug.get_train_pipeline())
    val_ds = TalcDataset([full_ds.image_paths[i] for i in val_idx], [full_ds.mask_paths[i] for i in val_idx],
                         transform=aug.get_val_pipeline())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TalcSegmentationModel(logger=logger)
    criterion = TalcLoss(logger=logger)
    metrics = TalcMetrics(logger=logger)
    best_iou = 0.0

    logger.info("Phase 1: Warmup")
    model.freeze_encoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = SegmentationTrainer(model, optimizer, criterion, device, metrics, logger)
    for e in range(1, warmup + 1):
        trainer.run_epoch(train_loader, e, True)
        _, vm = trainer.run_epoch(val_loader, e, False)
        logger.info(f"Warmup {e} | Val F1: {vm['f1']}")


    logger.info("Phase 2: Fine-Tuning")
    model.unfreeze_encoder()
    trainer.optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for e in range(warmup + 1, warmup + fine_tune + 1):
        trainer.run_epoch(train_loader, e, True)
        _, vm = trainer.run_epoch(val_loader, e, False)
        logger.info(f"Warmup {e} | Val F1: {vm['f1']}")
        if vm['iou'] > best_iou:
            best_iou = vm['iou']
            model.save_weights("weights/best_talc_model.pth")
            logger.info("New best model saved")


if __name__ == "__main__":
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    DATA_ROOT = os.path.join(project_root, "data")

    if not os.path.exists(DATA_ROOT):
        logger.info(f"Error Data folder not found at path: {DATA_ROOT}")
    else:
        train_pipeline(
            data_root=DATA_ROOT,
            warmup=10,
            fine_tune=40,
            batch_size=4
        )