# backend/inference.py

import os
import torch
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from PIL import Image

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# base paths
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 1) Image encoder (EfficientNet pretrained with SimCLR)
b0 = models.efficientnet_b0(weights=None)
img_enc = torch.nn.Sequential(b0.features, b0.avgpool).to(DEVICE).eval()
img_enc.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "effnet_simclr.pth"), map_location=DEVICE),
    strict=False
)

# 2) Tokenizer from HuggingFace Hub
tok = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# 3) Text encoder (fine-tuned weights locally)
txt_enc = AutoModel.from_pretrained(
    os.path.join(MODEL_DIR, "text_ckpt")
).to(DEVICE).eval()

# 4) Fusion head: projection + MLP
ckpt = torch.load(os.path.join(MODEL_DIR, "fusion_bundle.pt"), map_location=DEVICE)
img_proj = torch.nn.Linear(1280, 256, bias=False).to(DEVICE)
txt_proj = torch.nn.Linear( 768, 256, bias=False).to(DEVICE)

class FusionMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = torch.nn.Linear(512, 512)
        self.act  = torch.nn.ReLU(True)
        self.drop = torch.nn.Dropout(0.25)
        self.fc2  = torch.nn.Linear(512, 5)

    def forward(self, u):
        return self.fc2(self.drop(self.act(self.fc1(u))))

fusion = FusionMLP().to(DEVICE)
img_proj.load_state_dict(ckpt["img_proj"])
txt_proj.load_state_dict(ckpt["txt_proj"])
fusion.load_state_dict(ckpt["fusion"])
for m in (img_proj, txt_proj, fusion):
    m.eval()

LABELS = ["Fungal Infection", "Vitiligo", "Psoriasis", "Impetigo", "Urticaria"]

@torch.no_grad()
def predict(image_path: str = None, text: str = "") -> dict:
    # Image branch
    if image_path:
        from torchvision import transforms as T
        tf = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)
        ])
        img = Image.open(image_path).convert("RGB")
        img = tf(img).unsqueeze(0).to(DEVICE)
        zi  = img_proj(torch.flatten(img_enc(img), 1))
    else:
        zi = torch.zeros(1, 256, device=DEVICE)

    # Text branch
    if text.strip():
        enc = tok(text, return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(DEVICE)
        wt = txt_proj(txt_enc(**enc).last_hidden_state[:, 0])
    else:
        wt = torch.zeros(1, 256, device=DEVICE)

    # Fuse + classify
    logits = fusion(torch.cat([zi, wt], dim=1))
    probs  = torch.softmax(logits, dim=1).cpu().tolist()[0]
    return {lab: round(p, 4) for lab, p in zip(LABELS, probs)}
