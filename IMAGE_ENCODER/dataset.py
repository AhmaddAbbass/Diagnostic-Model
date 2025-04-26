# dataset + loaders (image‐only and img_text splits)
import random
import pandas as pd
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
from utils import clean_label

ROOT = Path("/content/drive/MyDrive/derm-mmmodal/final_divided")

class ImgDS(Dataset):
    def __init__(self, paths, labels, tfm=None):
        self.paths, self.labels, self.tfm = paths, labels, tfm
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.tfm: img = self.tfm(img)
        return img, self.labels[idx]

class TwoCrop:
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, x):   return self.tfm(x), self.tfm(x)

class SimCLRDS(Dataset):
    def __init__(self, base: ImgDS, two_crop: TwoCrop):
        self.base, self.two_crop = base, two_crop
    def __len__(self):  return len(self.base)
    def __getitem__(self, idx):
        img, lbl = self.base[idx]
        return (*self.two_crop(img), lbl)

def collect_split(split, focus=None, cap=None):
    paths, labels = [], []
    # image_only
    for d in (ROOT/"image_only"/split).iterdir():
        if not d.is_dir(): continue
        cls = clean_label(d.name)
        if focus and cls not in focus: continue
        files = list(d.iterdir())
        random.shuffle(files)
        take  = files[:cap] if cap else files
        paths += take; labels += [cls]*len(take)
    # img_text
    for xl in (ROOT/"img_text"/split).glob("*.xlsx"):
        df = pd.read_excel(xl)
        col = "label" if "label" in df.columns else "cat"
        df[col] = df[col].apply(clean_label)
        if focus: df = df[df[col].isin(focus)]
        if cap:   df = df.groupby(col).head(cap)
        for _,r in df.iterrows():
            f = xl.parent/"images"/r["image"]
            if f.exists():
                paths.append(f)
                labels.append(r[col])
    return paths, labels

def build_loaders(wanted_labels=None, img_sz=224, batch=64, cap=None, workers=2):
    focus = set(wanted_labels) if wanted_labels else None
    tr_p,tr_l = collect_split("train", focus, cap)
    va_p,va_l = collect_split("val",   focus, None)
    te_p,te_l = collect_split("test",  focus, None)

    classes = sorted({*tr_l,*va_l,*te_l})
    to_idx  = {c:i for i,c in enumerate(classes)}
    tr_l = [to_idx[l] for l in tr_l]
    va_l = [to_idx[l] for l in va_l]
    te_l = [to_idx[l] for l in te_l]

    norm   = T.Normalize([0.5]*3, [0.5]*3)
    aug    = T.Compose([
        T.RandomResizedCrop(img_sz, scale=(0.08,1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
        T.RandomGrayscale(0.2),
        T.GaussianBlur(23, sigma=(0.1,2.0)),
        T.RandomSolarize(128, p=0.2),
        T.ToTensor(), norm
    ])
    eval_t = T.Compose([
        T.Resize(int(img_sz*256/224)),
        T.CenterCrop(img_sz),
        T.ToTensor(), norm
    ])

    β, cnt = 0.9999, Counter(tr_l)
    eff    = {c:(1-β**cnt[c])/(1-β) for c in cnt}
    wts    = [1/eff[l] for l in tr_l]
    sampler= WeightedRandomSampler(wts, len(wts), replacement=True)

    sim_ds   = SimCLRDS(ImgDS(tr_p, tr_l, None), TwoCrop(aug))
    probe_ds = ImgDS(tr_p, tr_l, eval_t)
    val_ds   = ImgDS(va_p, va_l, eval_t)
    test_ds  = ImgDS(te_p, te_l, eval_t)

    sim_dl   = DataLoader(sim_ds,   batch_size=batch, sampler=sampler,
                          num_workers=workers, drop_last=True, pin_memory=True)
    probe_dl = DataLoader(probe_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)

    return sim_dl, probe_dl, val_dl, test_dl, classes
