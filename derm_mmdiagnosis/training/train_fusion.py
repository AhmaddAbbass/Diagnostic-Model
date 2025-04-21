# training/train_fusion.py
"""
train_fusion.py
---------------
(Section “End‑to‑End Pipeline” – Stage 3)
Loads pre‑trained f_i and f_t, applies modality dropout, mixes real+synthetic
pairs, and trains the cross‑attention fusion head end‑to‑end.
"""
