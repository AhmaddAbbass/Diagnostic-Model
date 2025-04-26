import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNetB0(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        b0 = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        self.features, self.pool = b0.features, b0.avgpool

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)

class SimCLRModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone  = EfficientNetB0(pretrained=True)
        self.projector = nn.Sequential(
            nn.Linear(1280,1280,bias=False),
            nn.BatchNorm1d(1280), nn.ReLU(inplace=True),
            nn.Linear(1280,proj_dim)
        )

    def forward(self, x):
        v = self.backbone(x)
        z = F.normalize(self.projector(v), dim=1)
        return v, z

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2],0)
        sim = (z @ z.T)/self.t
        mask = torch.eye(2*B, device=sim.device).bool()
        sim = sim.masked_fill(mask, float("-inf"))
        targets = (torch.arange(2*B, device=sim.device)+B)%(2*B)
        return F.cross_entropy(sim, targets)
