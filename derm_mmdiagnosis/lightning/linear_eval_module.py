import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class DermLinearEval(pl.LightningModule):
    """
    Stage 2 Linear Evaluation:
    - Freezes the image encoder from SimCLR.
    - Trains a single Linear layer on top of frozen 1280‑d features.
    """
    def __init__(self, encoder: nn.Module, n_classes: int, lr: float = 1e-3):
        super().__init__()
        # Freeze encoder parameters
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder    = encoder
        # Linear head
        self.classifier = nn.Linear(1280, n_classes)
        # Accuracy metric
        self.accuracy   = MulticlassAccuracy(num_classes=n_classes)
        # Learning rate
        self.lr         = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract frozen features
        feats, _ = self.encoder(x)      # feats: (B, 1280)
        # Apply linear classifier
        return self.classifier(feats)   # logits: (B, n_classes)

    def training_step(self, batch, batch_idx):
        x, y    = batch                 # x: images, y: labels
        logits  = self(x)               # forward pass
        loss    = F.cross_entropy(logits, y)
        acc     = self.accuracy(logits.softmax(dim=-1), y)
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc',  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y    = batch
        logits  = self(x)
        loss    = F.cross_entropy(logits, y)
        acc     = self.accuracy(logits.softmax(dim=-1), y)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc',  acc,  prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y    = batch
        logits  = self(x)
        acc     = self.accuracy(logits.softmax(dim=-1), y)
        self.log('test/acc', acc, prog_bar=True)

    def configure_optimizers(self):
        # Only the linear head's parameters will be updated
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
