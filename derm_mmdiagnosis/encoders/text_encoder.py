# encoders/text_encoder.py
"""
text_encoder.py
---------------
(Section “Unimodal Encoders (f_i, f_t)” – Text Branch)
Defines the text encoder (e.g. BioClinicalBERT/PubMedBERT or med‑tuned LLaMA),
with optional classification head.  
Fine‑tuned on SCIN and text‑only sets to produce f_t(T).
"""
