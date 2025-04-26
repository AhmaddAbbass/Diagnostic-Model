import torch
import pandas as pd
from tqdm import tqdm
from torch import nn

from dataset  import build_loaders
from model    import SimCLRModel
from constants import EPOCHS, CAP_PER_CLASS, BATCH_SIZE, DEVICE

def main():
    _, _, _, test_dl, classes = build_loaders(
        wanted_labels=None,
        img_sz=224,
        batch=BATCH_SIZE,
        cap=CAP_PER_CLASS,
        workers=2
    )

    backbone = SimCLRModel(proj_dim=128).to(DEVICE)
    backbone.load_state_dict(torch.load(f"checkpoints_final/pretrain_ep{EPOCHS:03d}.pth"))
    backbone.eval()

    probe = nn.Linear(1280, len(classes)).to(DEVICE)
    probe.load_state_dict(torch.load("checkpoints_final/best_probe.pth"))
    probe.eval()

    records = []
    test_ds = test_dl.dataset  # ImgDS

    for idx, (img, label) in enumerate(tqdm(test_ds, desc="Evaluating")):
        x = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats,_ = backbone(x)
            logits  = probe(feats)
            probs   = torch.softmax(logits,1).cpu().tolist()[0]

        rec = {"path":str(test_ds.paths[idx]), "true_label":classes[label]}
        rec.update({cls: p for cls,p in zip(classes, probs)})
        records.append(rec)

    pd.DataFrame(records).to_csv("test_predictions.csv", index=False)
    print("Saved test_predictions.csv")

if __name__=="__main__":
    main()
