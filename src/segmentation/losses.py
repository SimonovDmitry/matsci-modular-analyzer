import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import logging

class TalcLoss(nn.Module):
    def __init__(self, logger=None):
        super().__init__()
        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, from_logits=True)
        self.focal = smp.losses.FocalLoss(mode='binary')

    def forward(self, logits, targets):
        return 0.7 * self.tversky(logits, targets) + 0.3 * self.focal(logits, targets)