import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from models.image_encoder.derm_efficientnet import DermEfficientNet
from models.text_encoder import TextEncoder
from models.fusion_heads import FUSION_HEADS

class FusionModule(pl.LightningModule):
    """
    Stage 4: End‑to‑end multimodal fusion training over ⟨I,T⟩.
    """
    def __init__(self, hparams, vocab_size, num_classes):
        super().__init__()
        self.save_hyperparameters(hparams)
        # (a) unimodal encoders
        self.img_enc = DermEfficientNet(num_classes=0)
        self.txt_enc = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=hparams.model.embedding_dim,
            hidden_dim=hparams.model.hidden_dim,
            num_layers=hparams.model.num_layers,
            dropout=hparams.model.dropout,
        )
        # (b) fusion head
        FusionCls = FUSION_HEADS[hparams.model.fusion_head]
        self.fusion = FusionCls(
            dim_i=1280,
            dim_t=hparams.model.hidden_dim,
            hidden=hparams.model.hidden_dim,
        )
        # (c) final classifier
        self.classifier = torch.nn.Linear(
            hparams.model.hidden_dim, num_classes
        )
        self.accuracy   = MulticlassAccuracy(num_classes=num_classes)
        self.lr         = hparams.optimizer.lr
        # InfoNCE temperature
        self.temperature = hparams.loss.lambda_nce

    def info_nce_loss(self, vi, vt):
        # same as in simclr_module
        zi, zj = F.normalize(vi, dim=1), F.normalize(vt, dim=1)
        sim = torch.matmul(zi, zj.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def forward(self, img, input_ids, attention_mask):
        vi,_ = self.img_enc(img)             # (B,1280)
        vt    = self.txt_enc(input_ids, attention_mask)  # (B,H)
        fused = self.fusion(vi, vt)          # (B,H)
        return self.classifier(fused)        # (B,num_classes)

    def training_step(self, batch, _):
        imgs = batch["image"]
        ids  = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(imgs, ids, mask)
        loss_ce = F.cross_entropy(logits, labels)

        # contrastive alignment loss
        vi,_ = self.img_enc(imgs)
        vt    = self.txt_enc(ids, mask)
        loss_nce = self.info_nce_loss(vi, vt)

        loss = (
            self.hparams.loss.lambda_ce  * loss_ce +
            self.hparams.loss.lambda_nce* loss_nce
        )
        acc = self.accuracy(logits.softmax(1), labels)

        self.log("train/loss", loss,   prog_bar=True)
        self.log("train/acc",  acc,    prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["image"], batch["input_ids"], batch["attention_mask"])
        loss   = F.cross_entropy(logits, batch["labels"])
        acc    = self.accuracy(logits.softmax(1), batch["labels"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc",  acc,  prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
