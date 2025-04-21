# fusion/gmu.py
"""
gmu.py
------
(Section “Fusion Strategies” – Mid)
Implements the Gated Multimodal Unit:
  g = σ(W₁v + W₂w),  h = g ⊙ v + (1–g) ⊙ w,
with KL regularization on g.  
Captures modality importance dynamically.
"""
