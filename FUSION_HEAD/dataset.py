import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image
class MultiModalDS(Dataset):
    """
    Combines paired, image-only, text-only, and synthetic samples.
    """
    def __init__(self, split:str, tfm):
        self.rows = []
        root = Path(BASE,"final_divided")
        #   -------- img_text (paired) --------
        xls = root/"img_text"/f"img_text_{split}.xlsx"
        if xls.exists():
            df = pd.read_excel(xls, engine="openpyxl")
            df.label = df.label.str.strip().str.title()
            df = df[df.label.isin(LABELS)]
            img_dir = root/"img_text"/split/"images"
            for _,r in df.iterrows():
                p = img_dir/r["image"]
                if p.exists():
                    self.rows.append((p, str(r["text"]), cls2id[r.label], True,  True))
        #   -------- image_only --------
        io_dir = root/"image_only"/split
        if io_dir.exists():
            for lab_dir in io_dir.iterdir():
                lab = lab_dir.name.strip().title()
                if lab not in LABELS: continue
                for p in lab_dir.glob("*.png"):
                    self.rows.append((p, "", cls2id[lab], True,  False))
        #   -------- text_only --------
        to_dir = root/"text_only"/split
        if to_dir.exists():
            for xl in to_dir.glob("text_only_*.xlsx"):
                lab = xl.stem.split("_",2)[-1].title()
                if lab not in LABELS: continue
                df = pd.read_excel(xl, engine="openpyxl")
                for _,r in df.iterrows():
                    self.rows.append((None, str(r["text"]), cls2id[lab], False, True))
        #   -------- synthetic --------
        syn_dir = root/"synthetic"/split
        if syn_dir.exists():
            for xl in syn_dir.glob("synthetic_*.xlsx"):
                lab = xl.stem.split("_",1)[-1].title()
                if lab not in LABELS: continue
                df = pd.read_excel(xl, engine="openpyxl")
                for _,r in df.iterrows():
                    self.rows.append((None, str(r["text"]), cls2id[lab], False, True))
        assert self.rows, f"No data for split '{split}'"
        self.tfm = tfm
        print(f"{split}: {len(self.rows):,} rows")

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        p, txt, y, has_img, has_txt = self.rows[idx]
        if has_img:
            img = Image.open(p).convert("RGB"); img = self.tfm(img)
        else:
            img = torch.zeros(3, IMG_SZ, IMG_SZ)
        return {"pixel": img, "text": txt if has_txt else "",
                "label": y, "has_img": has_img, "has_txt": has_txt}
