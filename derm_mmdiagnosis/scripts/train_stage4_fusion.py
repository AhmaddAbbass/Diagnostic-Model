#!/usr/bin/env python3
import os
import glob
import argparse
import yaml

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from derm_mmdiagnosis.data.datasets import FusionDataset  # you’d implement similarly
from derm_mmdiagnosis.transforms.image import eval_transforms
from derm_mmdiagnosis.transforms.text import get_tokenizer
from derm_mmdiagnosis.lightning.fusion_module import FusionModule

def main():
    parser = argparse.ArgumentParser("Fusion Training")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to configs/04_fusion.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest fusion checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Tokenizer
    tokenizer = get_tokenizer(
        model_name=cfg["model"]["tokenizer_name"],
        max_len=cfg["model"]["max_seq_len"]
    )

    # Dataset & DataLoader
    ds = FusionDataset(
        cfg["data"]["paired_excel"],
        image_transform=eval_transforms,
        text_tokenizer=tokenizer,
        real_synthetic_ratio=cfg["data"]["real_synthetic_ratio"]
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers",4),
        pin_memory=True
    )

    # Callbacks
    ckpt_dir = cfg["trainer"].get("ckpt_dir", "checkpoints/fusion")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="fusion-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc", mode="max", save_top_k=1
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
            model = FusionModule.load_from_checkpoint(
                latest, hparams=cfg,
                vocab_size=tokenizer.vocab_size,
                num_classes=cfg["data"]["num_classes"]
            )
        else:
            print("No checkpoint found, starting fresh.")
            model = FusionModule(
                hparams=cfg,
                vocab_size=tokenizer.vocab_size,
                num_classes=cfg["data"]["num_classes"]
            )
    else:
        print("Starting training from scratch.")
        model = FusionModule(
            hparams=cfg,
            vocab_size=tokenizer.vocab_size,
            num_classes=cfg["data"]["num_classes"]
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        gpus=cfg["trainer"].get("gpus", 1),
        precision=cfg["trainer"].get("precision", 16),
        callbacks=[ckpt_cb, early_stop],
        default_root_dir=ckpt_dir
    )

    trainer.fit(model, dl)
    print("Best checkpoint saved at:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
