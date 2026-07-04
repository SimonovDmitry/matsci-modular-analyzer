import segmentation_models_pytorch as smp
import torch
import logging


class TalcMetrics:

    def __init__(self, threshold=0.5, logger=None):
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
        self.logger.info(f"TalcMetrics initialized with threshold: {self.threshold}")

    def reset(self):
        self.running_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.batch_count = 0

    def update(self, logits, targets):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        tp, fp, fn, tn = smp.metrics.get_stats(logits, targets.long(), mode='binary', threshold=self.threshold)
        self.running_stats["tp"] += torch.sum(tp).item()
        self.running_stats["fp"] += torch.sum(fp).item()
        self.running_stats["fn"] += torch.sum(fn).item()
        self.running_stats["tn"] += torch.sum(tn).item()
        self.batch_count += 1

    def compute_epoch_metrics(self):
        tp = self.running_stats["tp"]
        fp = self.running_stats["fp"]
        fn = self.running_stats["fn"]

        iou = tp / (tp + fp + fn + 1e-7)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        return {
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }