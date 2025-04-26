import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import collate, save_checkpoint
from model import EffNet, FusionMLP, TextEncoder

# Hyperparameters and other initialization
EPOCHS = 20
BATCH = 64
LR = 1e-4
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_dl, val_dl, test_dl = prepare_data()  # Assuming you have your DataLoader setup here

# Models
img_enc = EffNet().to(DEVICE)
txt_enc = TextEncoder().to(DEVICE)
fusion = FusionMLP().to(DEVICE)

# Optimizer
opt = torch.optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()) + list(fusion.parameters()), lr=LR)

# Training Loop
best_val = 0.0
for ep in range(1, EPOCHS+1):
    fusion.train(); img_enc.train(); txt_enc.train()
    for batch in train_dl:
        imgs = batch["images"].to(DEVICE)
        ids = batch["input_ids"].to(DEVICE)
        msk = batch["attention_mask"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

        with torch.no_grad():
            z_raw = img_enc(imgs)
            w_raw = txt_enc.encode(ids, msk)

        z = img_proj(z_raw)
        w = txt_proj(w_raw)

        z[batch["has_img"]==0] = 0
        w[batch["has_txt"]==0] = 0

        loss = F.cross_entropy(fusion(torch.cat([z, w], 1)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Validation
    fusion.eval()
    correct = tot = 0
    with torch.no_grad():
        for b in val_dl:
            z = img_proj(img_enc(b["images"].to(DEVICE)))
            w = txt_proj(txt_enc.encode(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)))
            z[b["has_img"] == 0] = 0
            w[b["has_txt"] == 0] = 0
            pred = fusion(torch.cat([z, w], 1)).argmax(1)
            correct += (pred == b["labels"].to(DEVICE)).sum().item()
            tot += pred.size(0)
    acc = 100 * correct / tot
    print(f"Epoch {ep}/{EPOCHS}  val {acc:.2f}%")
    
    if acc > best_val:
        best_val = acc
        save_checkpoint(fusion, "best_mlp.pt")

# Final test
test_accuracy(test_dl, fusion)
