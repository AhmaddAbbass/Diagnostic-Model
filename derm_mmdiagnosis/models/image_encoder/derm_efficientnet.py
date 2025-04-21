import torch.nn as nn
from .mbconv import MBConv

class DermEfficientNet(nn.Module):
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
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Blocks
        in_ch = 32
        total = sum(r for *_,r,*_ in cfgs)
        blocks = []
        b_i = 0
        for exp, out_ch, rep, k, s in cfgs:
            for i in range(rep):
                stride = s if i==0 else 1
                drop   = 0.2 * b_i/total
                blocks.append(MBConv(in_ch,out_ch,exp,k,stride,drop_connect=drop))
                in_ch = out_ch
                b_i += 1
        self.blocks = nn.Sequential(*blocks)
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch,1280,1,bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = (nn.Linear(1280,num_classes)
                           if num_classes>0 else None)

        # init weights omitted for brevity

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        feat = self.head(x)
        return (feat, self.classifier(feat)) if self.classifier else (feat,None)
