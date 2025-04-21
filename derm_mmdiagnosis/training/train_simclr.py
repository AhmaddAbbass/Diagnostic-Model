# training/train_simclr.py
"""
train_simclr.py
---------------
(Section “End‑to‑End Pipeline” – Stage 1)
Pre‑trains DermEfficientNet encoder with InfoNCE on the priority image‑only subset.
Uses ContrastiveImageFolder; checkpoints best model on contrastive loss.
"""







import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets
from torch.utils.data import DataLoader
from data.contrastive_dataset import FullContrastiveDataset
from utils.transforms import contrast_transforms
from models.contrastive_model import DermContrastiveModel

# Constants or import from config.py
BASE_DIR   = '/content/drive/.../final/image_only_divided'
CKPT_DIR   = '/content/drive/.../checkpoints'
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 3e-4
NUM_WORKERS= 4

os.makedirs(CKPT_DIR, exist_ok=True)

# Create dataset + loader
raw_folder = datasets.ImageFolder(f"{BASE_DIR}/train", transform=None)
contrast_ds = FullContrastiveDataset(raw_folder, contrast_transforms)
contrast_loader = DataLoader(
    contrast_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, drop_last=True, pin_memory=True
)

# Model + checkpoint
ckpt_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename='simclr-{epoch:02d}-{train/contrastive_loss:.2f}',
    monitor='train/contrastive_loss',
    mode='min',
    save_top_k=1
)
trainer = pl.Trainer(
    accelerator="gpu", devices=1, precision="16-mixed",
    max_epochs=EPOCHS, callbacks=[ckpt_cb], log_every_n_steps=20
)

model = DermContrastiveModel(lr=LR)
trainer.fit(model, contrast_loader)
