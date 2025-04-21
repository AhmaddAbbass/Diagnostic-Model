#!/usr/bin/env python3
import os
import glob
import argparse
import yaml

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from derm_mmdiagnosis.data.datasets import TextClassificationDataset
from derm_mmdiagnosis.transforms.text import get_tokenizer
from derm_mmdiagnosis.lightning.text_module import TextModule

def main():
    parser = argparse.ArgumentParser("Text Fine‑Tuning")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to configs/03_text_finetune.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest text checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Tokenizer + Datasets
    tokenizer = get_tokenizer(
        model_name=cfg["model"]["tokenizer_name"],
        max_len=cfg["model"]["max_seq_len"]
    )
    train_ds = TextClassificationDataset(
        cfg["data"]["scin_meta"], tokenizer
    )
    text_ds  = TextClassificationDataset(
        cfg["data"]["text_meta"], tokenizer
    )
    # combine or use separately; here we just train on SCIN then on text_ds...
    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=4)
    val_dl   = DataLoader(text_ds,  batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=4)

    # Callbacks
    ckpt_dir = cfg["trainer"].get("ckpt_dir", "checkpoints/text")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="text-{epoch:02d}-{val_acc:.4f}",
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
            model = TextModule.load_from_checkpoint(latest, hparams=cfg,
                                                    vocab_size=tokenizer.vocab_size,
                                                    num_classes=cfg["data"]["num_classes"])
        else:
            print("No checkpoint found, starting fresh.")
            model = TextModule(hparams=cfg,
                               vocab_size=tokenizer.vocab_size,
                               num_classes=cfg["data"]["num_classes"])
    else:
        print("Starting training from scratch.")
        model = TextModule(hparams=cfg,
                           vocab_size=tokenizer.vocab_size,
                           num_classes=cfg["data"]["num_classes"])

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        gpus=cfg["trainer"].get("gpus", 1),
        precision=cfg["trainer"].get("precision", 16),
        callbacks=[ckpt_cb, early_stop],
        default_root_dir=ckpt_dir,
    )

    trainer.fit(model, train_dl, val_dl)
    print("Best checkpoint saved at:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
