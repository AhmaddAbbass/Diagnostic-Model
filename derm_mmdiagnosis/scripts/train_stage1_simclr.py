#!/usr/bin/env python3
import os
import glob
import argparse
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from derm_mmdiagnosis.data.datasets import TwoViewImageFolder
from derm_mmdiagnosis.transforms.image import contrast_transforms
from derm_mmdiagnosis.lightning.simclr_module import SimCLRModule

def main():
    parser = argparse.ArgumentParser("SimCLR Training")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to configs/01_simclr.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # DataLoader
    ds = TwoViewImageFolder(
        cfg["data"]["train_dir"],
        contrast_transforms
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True
    )

    # Checkpoint callback
    ckpt_dir = cfg["trainer"].get("ckpt_dir", "checkpoints/simclr")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="simclr-{epoch:02d}-{train/contrastive_loss:.4f}",
        monitor="train/contrastive_loss",
        mode="min",
        save_top_k=1
    )

    # Instantiate or resume model
    if args.resume:
        all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if all_ckpts:
            latest = max(all_ckpts, key=os.path.getmtime)
            print(f"Resuming from {latest}")
            model = SimCLRModule.load_from_checkpoint(latest, hparams=cfg)
        else:
            print("No checkpoint found, starting fresh.")
            model = SimCLRModule(hparams=cfg)
    else:
        print("Starting training from scratch.")
        model = SimCLRModule(hparams=cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        gpus=cfg["trainer"].get("gpus", 1),
        precision=cfg["trainer"].get("precision", 16),
        callbacks=[ckpt_cb],
        default_root_dir=ckpt_dir,
    )

    # Fit
    trainer.fit(model, dl)
    print("Best checkpoint saved at:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
