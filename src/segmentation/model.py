import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import logging
import os


class TalcSegmentationModel(nn.Module):
    def __init__(self, encoder_name="efficientnet-b4", encoder_weights="imagenet", dropout=0.2, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.encoder_name = encoder_name

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            decoder_attention_type='scse'
        )

        if dropout > 0:
            self.model.segmentation_head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                self.model.segmentation_head
            )

        self._log_parameter_count()

    def forward(self, x):
        return self.model(x)

    def predict_mask(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float()
        return mask

    def _log_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Model stats: Total params {total_params:,}, Trainable {trainable_params:,}")

    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.logger.info(f"Encoder '{self.encoder_name}' is frozen. Training decoder only.")

    def unfreeze_encoder(self):

        for param in self.model.encoder.parameters():
            param.requires_grad = True
        self.logger.info(f"Encoder '{self.encoder_name}' is unfrozen. Entire network is ready for fine-tuning.")

    def save_weights(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.state_dict(), path)
            self.logger.info(f"Model weights saved successfully: {path}")
        except Exception as e:
            self.logger.error(f"Error occurred while saving weights: {e}")

    def load_weights(self, path, device="cpu"):
        if not os.path.exists(path):
            self.logger.error(f"Weights file not found: {path}")
            return False

        try:
            state_dict = torch.load(path, map_location=device)
            self.load_state_dict(state_dict)
            self.float()
            self.eval()
            self.logger.info(f"Weights loaded successfully from: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error occurred while loading weights: {e}")
            return False