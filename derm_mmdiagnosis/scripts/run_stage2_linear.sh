#!/usr/bin/env bash
python -m lightning.linear_eval_module \
  --config configs/02_linear_eval.yaml \
  --data-root /content/drive/MyDrive/derm-mmmodal/final/image_only_divided \
  --ckpt-dir /content/drive/MyDrive/derm-mmmodal/checkpoints
