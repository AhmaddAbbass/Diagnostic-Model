import pandas as pd
from pathlib import Path

# --- Option A: Load from your merged Excel file ---
merged_path = Path(__file__).parent / "merged_datasets_texts.xlsx"
df = pd.read_excel(merged_path, engine="openpyxl", dtype=str)

# --- Option B: Or, re-load & concatenate all your CSVs ---
# root = Path(__file__).parent / "datasets_texts"
# parts = []
# for f in root.glob("*.csv"):
#     parts.append(pd.read_csv(f, dtype=str))
# df = pd.concat(parts, ignore_index=True)

# Make sure the 'label' column exists
if 'label' not in df.columns:
    raise KeyError("No 'label' column found in your data!")

# Compute and print counts
label_counts = df['label'].value_counts(dropna=False)
print("Label counts:")
print(label_counts)

# (Optional) Save the counts to a CSV or Excel:
output = Path(__file__).parent / "label_counts.xlsx"
label_counts.to_frame(name="count").reset_index()\
            .rename(columns={'index':'label'})\
            .to_excel(output, index=False)
print(f"\n✅ Saved counts to {output}")
