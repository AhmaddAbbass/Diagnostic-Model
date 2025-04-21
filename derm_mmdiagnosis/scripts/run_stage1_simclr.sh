#!/usr/bin/env bash
python -m lightning.simclr_module \
  --config configs/01_simclr.yaml \
  --data-root /content/drive/MyDrive/derm-mmmodal/final/image_only_divided/train \
  --ckpt-dir /content/drive/MyDrive/derm-mmmodal/checkpoints
