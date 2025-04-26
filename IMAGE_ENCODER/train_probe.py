import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange, tqdm

from dataset  import build_loaders
from model    import SimCLRModel
from constants import EPOCHS, PROBE_EPOCHS, LR, CAP_PER_CLASS, BATCH_SIZE, DEVICE

def main():
    # load data
    _, probe_dl, val_dl, _, classes = build_loaders(
        wanted_labels=None,
        img_sz=224,
        batch=BATCH_SIZE,
        cap=CAP_PER_CLASS,
        workers=2
    )

    # load frozen backbone
    backbone = SimCLRModel(proj_dim=128).to(DEVICE)
    backbone.load_state_dict(torch.load(f"checkpoints_final/pretrain_ep{EPOCHS:03d}.pth"))
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False

    # linear probe
    probe = nn.Linear(1280, len(classes)).to(DEVICE)
    opt_p = torch.optim.AdamW(probe.parameters(), lr=LR)
    best  = 0.0

    for ep in trange(1, PROBE_EPOCHS+1, desc="Probe epochs"):
        probe.train()
        for x,y in tqdm(probe_dl, leave=False):
            x,y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                feats,_ = backbone(x)
            loss = F.cross_entropy(probe(feats), y)
            opt_p.zero_grad(); loss.backward(); opt_p.step()

        # validate
        probe.eval()
        correct=total=0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                preds = probe(backbone(x)[0]).argmax(1)
                correct += (preds==y).sum().item()
                total   += y.size(0)
        acc = 100*correct/total
        if acc>best:
            best=acc; torch.save(probe.state_dict(),"checkpoints_final/best_probe.pth")
        print(f"Epoch {ep}/{PROBE_EPOCHS} – Val Acc {acc:.2f}% (best {best:.2f}%)")

if __name__=="__main__":
    main()
