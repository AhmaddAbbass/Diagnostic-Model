# training/train_linear_full.py
"""
train_linear_full.py
--------------------
(Section “End‑to‑End Pipeline” – Stage 2 & 4)
Frozen‑encoder linear evaluation on the full dataset:
  1) Load best SimCLR checkpoint,
  2) Train linear head on train/val splits,
  3) Evaluate on test split,
  4) Save final encoder and linear head weights.
"""






import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.transforms import eval_transforms
from models.contrastive_model import DermContrastiveModel
from models.linear_eval_model import DermLinearEval

# Constants or import from config.py
BASE_DIR   = '/content/drive/.../final/image_only_divided'
CKPT_DIR   = '/content/drive/.../checkpoints'
BATCH_SIZE = 64
CONTR_LR   = 3e-4
EVAL_LR    = 1e-3
CONTR_EPOCHS = 20
EVAL_EPOCHS  = 10
NUM_WORKERS  = 4

# Load best SimCLR
best_simclr = os.path.join(CKPT_DIR, 'simclr-*.ckpt')  # or read from Trainer callback
simclr = DermContrastiveModel.load_from_checkpoint(best_simclr)
encoder = simclr.encoder

# DataLoaders
train_ds = datasets.ImageFolder(f'{BASE_DIR}/train', transform=eval_transforms)
val_ds   = datasets.ImageFolder(f'{BASE_DIR}/val',   transform=eval_transforms)
test_ds  = datasets.ImageFolder(f'{BASE_DIR}/test',  transform=eval_transforms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Linear Eval Module + Callbacks
ckpt_linear = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename='linear-{epoch:02d}-{val_acc:.2f}',
    monitor='val_acc',
    mode='max',
    save_top_k=1
)
early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=3)

lin_module = DermLinearEval(encoder, n_classes=len(train_ds.classes), lr=EVAL_LR)
trainer = pl.Trainer(
    accelerator="gpu", devices=1, precision="16-mixed",
    max_epochs=EVAL_EPOCHS, callbacks=[ckpt_linear, early_stop], log_every_n_steps=20
)
trainer.fit(lin_module, train_loader, val_loader)
trainer.test(lin_module, dataloaders=test_loader)

# Save weights
torch.save(encoder.state_dict(), os.path.join(CKPT_DIR, 'final_encoder.pth'))
torch.save(lin_module.state_dict(), os.path.join(CKPT_DIR, 'final_linear.pth'))
