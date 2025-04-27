import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import MultiModalDS, IMG_SZ, MAX_LEN  # <- new
from utils   import collate, save_checkpoint
from model   import EffNet, TextEncoder, FusionMLP

# ─── HYPERPARAMS ────────────────────────────────────────────────────────
EPOCHS, BATCH, LR = 20, 64, 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── DATA ────────────────────────────────────────────────────────────────
train_ds = MultiModalDS("train", train_tf())
val_ds   = MultiModalDS("val",   val_tf)
test_ds  = MultiModalDS("test",  val_tf)

lbls     = [r[2] for r in train_ds.rows]
wts      = 1. / torch.tensor(torch.bincount(torch.tensor(lbls)), dtype=torch.float)
sampler  = WeightedRandomSampler(wts[lbls], len(lbls), True)

train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                      collate_fn=collate, num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                      collate_fn=collate, num_workers=2)
test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                      collate_fn=collate, num_workers=2)

# ─── MODELS ──────────────────────────────────────────────────────────────
img_enc   = EffNet().to(DEVICE).eval()
txt_enc   = TextEncoder().to(DEVICE).eval()
img_proj  = torch.nn.Linear(img_enc.dim,  256, bias=False).to(DEVICE)
txt_proj  = torch.nn.Linear(txt_enc.bert.config.hidden_size, 256, bias=False).to(DEVICE)
fusion    = FusionMLP().to(DEVICE)

# optionally warm-start your fusion head if you have a checkpoint
try:
    fusion.load_state_dict(torch.load("best_mlp.pt", map_location=DEVICE))
    print("✔︎ loaded best_mlp.pt")
except FileNotFoundError:
    pass

opt = torch.optim.AdamW(
    list(img_proj.parameters())+
    list(txt_proj.parameters())+
    list(fusion.parameters()),
    lr=LR
)

# ─── TRAIN ───────────────────────────────────────────────────────────────
best_val = 0.0
for ep in range(1, EPOCHS+1):
    fusion.train()
    for b in train_dl:
        imgs, ids, msk, y = (b["images"].to(DEVICE),
                             b["input_ids"].to(DEVICE),
                             b["attention_mask"].to(DEVICE),
                             b["labels"].to(DEVICE))
        hi, ht = b["has_img"].to(DEVICE), b["has_txt"].to(DEVICE)

        with torch.no_grad():
            zi_raw = img_enc(imgs)
            wt_raw = txt_enc(ids, msk)

        zi = img_proj(zi_raw)
        wt = txt_proj(wt_raw)

        zi[hi==0] = 0
        wt[ht==0] = 0

        logits = fusion(torch.cat([zi, wt], dim=1))
        loss   = F.cross_entropy(logits, y)

        opt.zero_grad(); loss.backward(); opt.step()

    # validation
    fusion.eval(); correct=tot=0
    with torch.no_grad():
        for b in val_dl:
            zi  = img_proj(img_enc(b["images"].to(DEVICE)))
            wt  = txt_proj(txt_enc(b["input_ids"].to(DEVICE),
                                    b["attention_mask"].to(DEVICE)))
            zi[b["has_img"]==0] = 0
            wt[b["has_txt"]==0] = 0
            pred = fusion(torch.cat([zi, wt],1)).argmax(1)
            correct += (pred==b["labels"].to(DEVICE)).sum().item()
            tot     += pred.size(0)
    acc = 100 * correct / tot
    print(f"Epoch {ep}/{EPOCHS} val {acc:.2f}%")
    if acc > best_val:
        best_val = acc
        save_checkpoint(fusion, "best_mlp.pt")
        print("  ✓ new best saved")

# ─── TEST ────────────────────────────────────────────────────────────────
from utils import test_accuracy
test_accuracy(test_dl, fusion, DEVICE)
