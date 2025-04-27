import json, random
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image

# ─── CONFIG ─────────────────────────────────────────────────────────────
BASE    = "/content/drive/MyDrive/derm-mmmodal"
LABELS  = ["Fungal Infection","Vitiligo","Psoriasis","Impetigo","Urticaria"]
cls2id  = {c:i for i,c in enumerate(LABELS)}
IMG_SZ  = 224
MAX_LEN = 256
TOK     = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# ─── load your precomputed clusters & ECDF-weights ──────────────────────
clusters_by_label = json.load(open(Path(BASE)/"clusters_by_label.json"))
probs_by_label    = json.load(open(Path(BASE)/"probs_by_label.json"))

class MultiModalDS(Dataset):
    def __init__(self, split:str, tfm, pseudo_pair:bool=True):
        self.rows = []
        root = Path(BASE,"final_divided")

        # 1) true paired img_text
        xls = root/"img_text"/f"img_text_{split}.xlsx"
        if xls.exists():
            df = pd.read_excel(xls, engine="openpyxl")
            df.label = df.label.str.strip().str.title()
            df = df[df.label.isin(LABELS)]
            img_dir = root/"img_text"/split/"images"
            for _,r in df.iterrows():
                p = img_dir/r["image"]
                if p.exists():
                    self.rows.append((p, str(r["text"]), cls2id[r.label], True, True))

        # 2) image_only
        io_dir = root/"image_only"/split
        if io_dir.exists():
            for lab_dir in io_dir.iterdir():
                lab = lab_dir.name.strip().title()
                if lab not in LABELS: continue
                for p in lab_dir.glob("*.png"):
                    if pseudo_pair and lab in clusters_by_label:
                        # sample a text from one of the clusters by ECDF
                        probs = probs_by_label[lab]
                        idx   = random.choices(range(len(probs)), weights=probs, k=1)[0]
                        txt   = random.choice(clusters_by_label[lab][idx])
                        self.rows.append((p, txt, cls2id[lab], True, True))
                    else:
                        self.rows.append((p, "", cls2id[lab], True, False))

        # 3) text_only & synthetic (always unpaired)
        for mod in ("text_only","synthetic"):
            d = root/mod/split
            if not d.exists(): continue
            for xl in d.glob(f"{mod}_*.xlsx"):
                lab = xl.stem.split("_")[-1].title()
                if lab not in LABELS: continue
                df = pd.read_excel(xl, engine="openpyxl")
                for _,r in df.iterrows():
                    self.rows.append((None, str(r["text"]), cls2id[lab], False, True))

        assert self.rows, f"No data in split '{split}'"
        self.tfm = tfm
        print(f"{split}: {len(self.rows):,} samples")

    def __len__(self): 
        return len(self.rows)

    def __getitem__(self, i):
        p, txt, y, hi, ht = self.rows[i]
        if hi:
            img = Image.open(p).convert("RGB")
            img = self.tfm(img)
        else:
            img = torch.zeros(3, IMG_SZ, IMG_SZ)
        return {
            "pixel":   img,
            "text":    txt if ht else "",
            "label":   y,
            "has_img": hi,
            "has_txt": ht
        }
