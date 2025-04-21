# training/train_text.py
"""
train_text.py
-------------
(Section “End‑to‑End Pipeline” – Stage 2)
Fine‑tunes the text encoder (BioClinicalBERT or LLaMA‑based) on SCIN image–text
pairs plus text‑only data, using MLM+NSP or classification loss.
"""
