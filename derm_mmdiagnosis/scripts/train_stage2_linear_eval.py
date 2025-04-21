#!/usr/bin/env python3
import os
import glob
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from derm_mmdiagnosis.data.datasets import ClassificationDataset
from derm_mmdiagnosis.transforms.image import eval_transforms
from derm_mmdiagnosis.lightning.linear_eval_module import LinearEvalModule

def main():
    parser = argparse.ArgumentParser("Linear Evaluation")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to configs/02_linear_eval.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest linear checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # DataLoaders
    train_dl = DataLoader(
        ClassificationDataset(cfg["data"]["train_dir"], eval_transforms),
        batch_size=cfg["data"]["batch_size"],
        shuffle=True, num_workers=cfg["data"].get("num_workers", 4)
    )
    val_dl = DataLoader(
        ClassificationDataset(cfg["data"]["val_dir"], eval_transforms),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"].get("num_workers", 4)
    )
    test_dl = DataLoader(
        ClassificationDataset(cfg["data"]["test_dir"], eval_transforms),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"].get("num_workers", 4)
    )

    # Checkpoint & early stop
    ckpt_dir = cfg["trainer"].get("ckpt_dir", "checkpoints/linear_eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="linear-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max", save_top_k=1
    )
    early_stop = EarlyStopping(
        monitor="val_acc", mode="max", patience=cfg["trainer"]["early_stopping"]["patience"]
    )

    # Instantiate or resume
    if args.resume:
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if ckpts:
            latest = max(ckpts, key=os.path.getmtime)
            print(f"Resuming from {latest}")
            model = LinearEvalModule.load_from_checkpoint(latest, hparams=cfg)
        else:
            print("No checkpoint found, starting fresh.")
            model = LinearEvalModule(hparams=cfg)
    else:
        print("Starting training from scratch.")
        model = LinearEvalModule(hparams=cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        gpus=cfg["trainer"].get("gpus", 1),
        precision=cfg["trainer"].get("precision", 16),
        callbacks=[ckpt_cb, early_stop],
        default_root_dir=ckpt_dir,
    )

    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl)
    print("Best checkpoint saved at:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
