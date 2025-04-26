# build dataset summaries (saves two Excel files)
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from utils import clean_label

ROOT = Path("/content/drive/MyDrive/derm-mmmodal/final_divided")
OUT1 = ROOT / "dataset_summary.xlsx"
OUT2 = ROOT / "dataset_summary_full.xlsx"

def count_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts)

records, totals = [], defaultdict(Counter)

for split in ("train","val","test"):
    # 1️⃣ image_only
    for d in (ROOT/"image_only"/split).iterdir():
        if not d.is_dir(): continue
        lbl = clean_label(d.name)
        n   = count_images(d)
        records.append({"label":lbl,"split":split,"modality":"image_only","n":n})
        totals[lbl][f"image_only_{split}"] += n

    # 2️⃣ img_text
    for xl in (ROOT/"img_text"/split).glob("*.xlsx"):
        df = pd.read_excel(xl)
        col = "label" if "label" in df.columns else "cat"
        df[col] = df[col].apply(clean_label)
        for lbl, cnt in df[col].value_counts().items():
            records.append({"label":lbl,"split":split,"modality":"img_text","n":int(cnt)})
            totals[lbl][f"img_text_{split}"] += int(cnt)

    # 3️⃣ synthetic
    for xl in (ROOT/"synthetic"/split).glob("*.xlsx"):
        df = pd.read_excel(xl)
        df["label"] = df["label"].apply(clean_label)
        for lbl, cnt in df["label"].value_counts().items():
            records.append({"label":lbl,"split":split,"modality":"synthetic","n":int(cnt)})
            totals[lbl][f"synthetic_{split}"] += int(cnt)

    # 4️⃣ text_only
    for xl in (ROOT/"text_only"/split).glob("*.xlsx"):
        df = pd.read_excel(xl)
        df["label"] = df["label"].apply(clean_label)
        for lbl, cnt in df["label"].value_counts().items():
            records.append({"label":lbl,"split":split,"modality":"text_only","n":int(cnt)})
            totals[lbl][f"text_only_{split}"] += int(cnt)

# save
pd.DataFrame(records).to_excel(OUT2, index=False)
rows = []
for lbl, cnts in totals.items():
    row = {"label":lbl, **cnts, "total":sum(cnts.values())}
    rows.append(row)
(pd.DataFrame(rows)
   .fillna(0)
   .astype({"total":int})
   .sort_values("label")
   .to_excel(OUT1, index=False))
print("Saved summaries:", OUT1, OUT2)
