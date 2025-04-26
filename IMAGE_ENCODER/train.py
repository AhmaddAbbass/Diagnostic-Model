import os
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange, tqdm

from utils    import set_seed
from dataset  import build_loaders
from model    import SimCLRModel, InfoNCELoss
from constants import EPOCHS, LR, CAP_PER_CLASS, BATCH_SIZE, DEVICE

def main():
    set_seed(123)
    os.makedirs("checkpoints_final", exist_ok=True)

    sim_dl, _, _, _, classes = build_loaders(
        wanted_labels=None,
        img_sz=224,
        batch=BATCH_SIZE,
        cap=CAP_PER_CLASS,
        workers=2
    )

    model   = SimCLRModel(proj_dim=128).to(DEVICE)
    loss_fn = InfoNCELoss(temperature=0.2)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=EPOCHS * len(sim_dl))
    scaler  = GradScaler()

    for ep in trange(1, EPOCHS+1, desc="SimCLR epochs"):
        model.train()
        pbar = tqdm(sim_dl, leave=False, desc="  batches")
        for x1, x2, _ in pbar:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            opt.zero_grad()
            with autocast():
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss  = loss_fn(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        torch.save(model.state_dict(),
                   f"checkpoints_final/pretrain_ep{ep:03d}.pth")

if __name__=="__main__":
    main()
