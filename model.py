"""
simclr_efficientnet_b0.py   ·   v3  (classifier kept, SEBlock bug fixed)

• EfficientNet-B0 backbone with built-in classifier
• SimCLR wrapper that returns   v, z, logits(optional)
• Every line commented for newcomers
"""
##from google.colab import drive
##drive.mount('/content/drive')
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 0) Alias: Swish = nn.SiLU (fused activation already in PyTorch)
# ---------------------------------------------------------------------
Swish = nn.SiLU


# ---------------------------------------------------------------------
# 1) Squeeze-and-Excitation (channel attention)
# ---------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed = max(1, int(round(in_ch * se_ratio)))
        self.fc1 = nn.Conv2d(in_ch, squeezed, 1)
        self.fc2 = nn.Conv2d(squeezed, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)     # global squeeze
        s = self.fc1(s)                     # ↓ channel dim
        s = Swish()(s)                      # ⚠️ instantiate then call
        s = torch.sigmoid(self.fc2(s))      # ↑ back to C & gate
        return x * s                        # scale channels


# ---------------------------------------------------------------------
# 2) MBConv block (inverted bottleneck + SE)
# ---------------------------------------------------------------------
class MBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int,
        k: int,
        stride: int,
        se_ratio: float,
    ):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        mid = int(round(in_ch * expand))

        layers: list[nn.Module] = []

        # (a) 1×1 expand
        if expand != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                        nn.BatchNorm2d(mid), Swish()]

        # (b) depth-wise k×k conv
        layers += [nn.Conv2d(mid, mid, k, stride,
                             padding=k // 2,
                             groups=mid, bias=False),
                   nn.BatchNorm2d(mid), Swish()]

        # (c) Squeeze-and-Excitation
        layers.append(SEBlock(mid, se_ratio))

        # (d) 1×1 project
        layers += [nn.Conv2d(mid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        h = self.block(x)
        return h + x if self.use_res else h


# ---------------------------------------------------------------------
# 3) EfficientNet-B0 backbone (Image Encoder)
# ---------------------------------------------------------------------
class EfficientNetB0(nn.Module):
    """
    • extract_features(x) → 1280-D vector v
    • forward(x)          → logits if num_classes>0, else v
    """
    def __init__(self, num_classes: int = 0):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 3×3, stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
        )

        # MBConv config: (expand, k, out, repeats, stride)
        cfgs = [
            (1, 3,  16, 1, 1),
            (6, 3,  24, 2, 2),
            (6, 5,  40, 2, 2),
            (6, 3,  80, 3, 2),
            (6, 5, 112, 3, 1),
            (6, 5, 192, 4, 2),
            (6, 3, 320, 1, 1),
        ]

        blocks, in_ch = [], 32
        for exp, k, out, n, s in cfgs:
            for i in range(n):
                blocks.append(
                    MBConv(in_ch, out, exp, k,
                           stride=s if i == 0 else 1,
                           se_ratio=0.25)
                )
                in_ch = out
        self.blocks = nn.Sequential(*blocks)

        # Head conv 1×1 → 1280-D
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
        )

        # Pool + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280, num_classes)

    # ----- feature extractor used in all stages -----
    def extract_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        return self.pool(x).flatten(1)      # (B, 1280)

    # ----- standard forward -----
    def forward(self, x):
        feats = self.extract_features(x)
        if self.num_classes > 0:
            return self.classifier(self.dropout(feats))
        return feats                       # same as v


# ---------------------------------------------------------------------
# 4) SimCLR wrapper  (Stage-1 pre-training)
# ---------------------------------------------------------------------
class SimCLRModel(nn.Module):
    """
    Returns
    -------
    v : 1280-D global visual feature (feeds Fusion Head)
    z : proj-dim L2-normalised projection (InfoNCE loss)
    logits : optional image-only logits (for later fine-tuning)
    """
    def __init__(self, proj_dim=128, num_classes=0):
        super().__init__()
        self.backbone = EfficientNetB0(num_classes=num_classes)

        self.projector = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, proj_dim),
        )

        # Keep classifier frozen during SimCLR
        for p in self.backbone.classifier.parameters():
            p.requires_grad = False

    def enable_classifier_training(self):
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def forward(self, x, return_logits: bool = False):
        v = self.backbone.extract_features(x)      # 1280-D
        z = F.normalize(self.projector(v), dim=1)  # 128-D
        logits = (self.backbone.classifier(v)
                  if return_logits and self.backbone.num_classes > 0
                  else None)
        return v, z, logits
