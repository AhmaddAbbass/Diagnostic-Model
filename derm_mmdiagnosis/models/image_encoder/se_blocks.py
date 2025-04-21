import torch, torch.nn as nn

def swish(x): 
    return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(in_ch*se_ratio))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Conv2d(in_ch, reduced, 1)
        self.fc2     = nn.Conv2d(reduced, in_ch, 1)

    def forward(self, x):
        se = self.avgpool(x)
        se = swish(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se
