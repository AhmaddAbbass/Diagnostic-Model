import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image

class MultiModalDS(Dataset):
    def __init__(self, split: str, tfm):
        self.rows = []
        root = Path(BASE, "final_divided")
        xls = root / "img_text" / f"img_text_{split}.xlsx"
        if xls.exists():
            df = pd.read_excel(xls, engine="openpyxl")
            df.label = df.label.str.strip().str.title()
            df = df[df.label.isin(LABELS)]
            img_dir = root / "img_text" / split / "images"
            for _, r in df.iterrows():
                p = img_dir / r["image"]
                if p.exists():
                    self.rows.append((p, str(r["text"]), cls2id[r.label], True, True))
        # ... (other dataset logic for image_only, text_only, synthetic)
        
        self.tfm = tfm
        print(f"{split}: {len(self.rows):,} rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        p, txt, y, has_img, has_txt = self.rows[idx]
        if has_img:
            img = Image.open(p).convert("RGB")
            img = self.tfm(img)
        else:
            img = torch.zeros(3, IMG_SZ, IMG_SZ)
        return {"pixel": img, "text": txt if has_txt else "", "label": y, "has_img": has_img, "has_txt": has_txt}
