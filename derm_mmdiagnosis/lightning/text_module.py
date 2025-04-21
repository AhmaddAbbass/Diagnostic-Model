import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from models.text_encoder import TextEncoder

class TextModule(pl.LightningModule):
    """
    Stage 3: Text‐only fine‑tuning on SCIN + text‑only clinical descriptions.
    """
    def __init__(self, hparams, vocab_size, num_classes):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder    = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=hparams.model.embedding_dim,
            hidden_dim=hparams.model.hidden_dim,
            num_layers=hparams.model.num_layers,
            dropout=hparams.model.dropout,
        )
        self.classifier = torch.nn.Linear(
            hparams.model.hidden_dim, num_classes
        )
        self.accuracy   = MulticlassAccuracy(num_classes=num_classes)
        self.lr         = hparams.optimizer.lr

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        return self.classifier(features)

    def training_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss   = F.cross_entropy(logits, batch["labels"])
        acc    = self.accuracy(logits.softmax(1), batch["labels"])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc",  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss   = F.cross_entropy(logits, batch["labels"])
        acc    = self.accuracy(logits.softmax(1), batch["labels"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc",  acc,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
