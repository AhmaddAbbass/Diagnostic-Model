#!/usr/bin/env python
# list_unique_labels.py
#
# Collect unique labels from:
#   • scin_master.xlsx              (structured)
#   • datasets_texts/<disease>/*    (CSV or XLSX in each sub‑folder)
#
# Usage:
#   pip install pandas
#   python list_unique_labels.py
# ------------------------------------------------------

from pathlib import Path
import pandas as pd

ROOT      = Path(".")
SCIN_PATH = ROOT / "scin_master.xlsx"
TXT_ROOT  = ROOT / "datasets_texts"

def normalize(series: pd.Series) -> pd.Series:
    """lower‑case, trim, drop NaNs → str"""
    return series.astype(str).str.strip().str.lower().dropna()

# ---------- structured --------------------------------
scin = pd.read_excel(SCIN_PATH, usecols=["label"])
structured_labels = sorted(normalize(scin["label"]).unique())

# ---------- unstructured ------------------------------
u_labels = []

def read_labels(path: Path) -> list[str]:
    """Return a list of labels from a CSV or XLSX, robust to index column + tab."""
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, sep=None, engine="python")  # auto‑detect delimiter
        except Exception as e:
            print(f"[WARN] couldn't read {path.name}: {e}")
            return []
    else:  # .xlsx
        df = pd.read_excel(path)

    # Drop an index col like 'Unnamed: 0' if present
    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]

    # Find the first column whose name contains "label"
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if not label_cols:
        print(f"[WARN] no label column in {path.name}. cols={list(df.columns)}")
        return []

    return normalize(df[label_cols[0]]).tolist()

for disease_dir in TXT_ROOT.iterdir():
    if not disease_dir.is_dir():
        continue

    # prefer CSV, else XLSX
    files = list(disease_dir.glob("*.csv")) or list(disease_dir.glob("*.xlsx"))
    if not files:
        print(f"[WARN] no data file in {disease_dir}")
        continue

    u_labels.extend(read_labels(files[0]))

unstructured_labels = sorted(pd.unique(u_labels))

# ---------- print results -----------------------------
print("\nSTRUCTURED_LABELS:")
for lab in structured_labels:
    print("  •", lab)

print("\nUNSTRUCTURED_LABELS:")
for lab in unstructured_labels:
    print("  •", lab)

print(f"\nStructured count   : {len(structured_labels)}")
print(f"Unstructured count : {len(unstructured_labels)}")
print(f"Overlap            : {len(set(structured_labels) & set(unstructured_labels))}")
