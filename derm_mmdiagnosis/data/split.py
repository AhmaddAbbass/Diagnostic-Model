# data/split.py
"""
split.py
--------
(Section “Data & Preprocessing”)
Performs 70/15/15 stratified train/val/test splits on each modality
(image‑only, image–text, text‑only) while preserving per‑disease balance.
Outputs both a full split and the priority “final” subset.
"""
