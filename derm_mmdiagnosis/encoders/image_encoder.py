# encoders/image_encoder.py
"""
image_encoder.py
----------------
(Section “Unimodal Encoders (f_i, f_t)” – Image Branch)
Implements DermEfficientNet‑B0:
  • Swish activation, SEBlock, MBConv blocks
  • 1280‑dim feature head, optional classifier
Used both for SimCLR pre‑training and as f_i in fusion.
"""











import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
def swish(x):
    """Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(in_ch * se_ratio))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, in_ch, 1)
    def forward(self, x):
        se = self.avgpool(x)
        se = swish(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with SE and stochastic depth."""
    def __init__(self, in_ch, out_ch, expansion, kernel_size, stride,
                 se_ratio=0.25, drop_connect=0.2):
        super().__init__()
        self.has_residual = (in_ch == out_ch and stride == 1)
        mid_ch = in_ch * expansion
        # expansion
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False) if expansion != 1 else nn.Identity()
        self.bn0    = nn.BatchNorm2d(mid_ch) if expansion != 1 else nn.Identity()
        # depthwise
        self.dw = nn.Conv2d(mid_ch, mid_ch, kernel_size,
                            stride=stride, padding=kernel_size//2,
                            groups=mid_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        # SE
        self.se = SEBlock(mid_ch, se_ratio)
        # projection
        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn2     = nn.BatchNorm2d(out_ch)
        self.drop_connect = drop_connect
    def forward(self, x):
        out = swish(self.bn0(self.expand(x)))
        out = swish(self.bn1(self.dw(out)))
        out = self.se(out)
        out = self.bn2(self.project(out))
        if self.has_residual:
            if self.drop_connect and self.training:
                keep = 1 - self.drop_connect
                mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < keep
                out = out.div(keep) * mask
            out = out + x
        return out

# ── DermEfficientNet Definition ────────────────────────────────────────────
class DermEfficientNet(nn.Module):
    """
    EfficientNet-B0 variant:
    - Stem: initial Conv-BN-ReLU
    - Blocks: MBConv series
    - Head: conv1x1 → 1280 → pool → flatten
    - Optional classifier if num_classes>0
    """
    def __init__(self, num_classes=0):
        super().__init__()
        cfgs = [
            (1,16,1,3,1),(6,24,2,3,2),(6,40,2,5,2),
            (6,80,3,3,2),(6,112,3,5,1),(6,192,4,5,2),
            (6,320,1,3,1)
        ]
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        # Blocks
        in_ch = 32
        total = sum(r for _,_,r,_,_ in cfgs)
        b_i = 0
        blocks = []
        for exp, out_ch, rep, k, s in cfgs:
            for i in range(rep):
                stride = s if i==0 else 1
                drop = 0.2 * b_i / total
                blocks.append(MBConv(in_ch, out_ch, exp, k, stride, drop_connect=drop))
                in_ch = out_ch
                b_i += 1
        self.blocks = nn.Sequential(*blocks)
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch,1280,1,bias=False),
            nn.BatchNorm2d(1280), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        # Classifier (if requested)
        self.classifier = nn.Linear(1280, num_classes) if num_classes>0 else None
        # Init
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        feat = self.head(x)
        return (feat, self.classifier(feat)) if self.classifier else (feat, None)

# ── DermContrastiveModel Lightning Module ─────────────────────────────────
class DermContrastiveModel(pl.LightningModule):
    """
    SimCLR-style contrastive learner:
    - Encoder: DermEfficientNet(no classifier)
    - Projector: 2-layer MLP → 128-d
    - InfoNCE loss
    """
    def __init__(self, lr=3e-4):
        super().__init__()
        self.encoder = DermEfficientNet(num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(1280,1280), nn.ReLU(inplace=True),
            nn.Linear(1280,128)
        )
        self.lr = lr
    def info_nce_loss(self, z1, z2, T=0.5):
        z1,z2 = F.normalize(z1,1), F.normalize(z2,1)
        N = z1.size(0)
        sim = torch.matmul(torch.cat([z1,z2],0),
                           torch.cat([z1,z2],0).T) / T
        mask = torch.eye(2*N, device=sim.device).bool()
        # Change -9e15 to a smaller negative value within the range of float16
        sim = sim.masked_fill(mask, -65504.0)  
        pos = torch.cat([torch.diag(sim,N), torch.diag(sim,-N)],0)
        neg = sim[~mask].view(2*N,-1)
        logits = torch.cat([pos.unsqueeze(1), neg],1)
        labels = torch.zeros(2*N, dtype=torch.long, device=sim.device)
        return F.cross_entropy(logits, labels)
    def training_step(self, batch, batch_idx):
        x1,x2 = batch
        h1,_ = self.encoder(x1); h2,_ = self.encoder(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        loss = self.info_nce_loss(z1,z2)
        self.log('train/contrastive_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ── Contrastive Dataset ─────────────────────────────────────────────────
class ContrastiveImageFolder(Dataset):
    """
    Wraps ImageFolder to return two augmentations per image.
    """
    def __init__(self, root, transform):
        self.folder = datasets.ImageFolder(root, transform=None)
        self.transform = transform
    def __len__(self):
        return len(self.folder)
    def __getitem__(self, idx):
        img, _ = self.folder[idx]   # discard label
        return self.transform(img), self.transform(img)
# ── DermLinearEval Lightning Module ──────────────────────────────────────
class DermLinearEval(pl.LightningModule):
    """
    Freeze contrastive encoder, train a linear classifier on features.
    """
    def __init__(self, encoder: DermEfficientNet, n_classes: int, lr=1e-3):
        super().__init__()
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder
        self.classifier = nn.Linear(1280, n_classes)
        self.accuracy = MulticlassAccuracy(num_classes=n_classes)
        self.lr = lr
    def forward(self, x):
        feats, _ = self.encoder(x)
        return self.classifier(feats)
    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.softmax(1), y)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc',  acc,  prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc  = self.accuracy(logits.softmax(1), y)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc',  acc,  prog_bar=True)
    def test_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        acc  = self.accuracy(logits.softmax(1), y)
        self.log('test/acc', acc, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)