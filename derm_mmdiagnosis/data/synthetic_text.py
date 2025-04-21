# data/synthetic_text.py
"""
synthetic_text.py
-----------------
(Section “Multimodal Pipeline: Detailed Implementation Road‑Map”)
Generates synthetic clinical reports for DermNet images via:
  1) FAISS nearest‑neighbour bootstrap,
  2) LLM refinement (GPT‑4/LLaMA prompt),
  3) SNOMED CT ontology injection.
Produces pseudo‑paired (I,~T, y) data for fusion training.
"""
