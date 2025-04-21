import torch, torch.nn as nn
from .se_block import SEBlock, swish

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, k, s,
                 se_ratio=0.25, drop_connect=0.2):
        super().__init__()
        self.has_residual = (in_ch==out_ch and s==1)
        mid_ch = in_ch*expansion

        self.expand = (nn.Conv2d(in_ch,mid_ch,1,bias=False)
                       if expansion!=1 else nn.Identity())
        self.bn0    = (nn.BatchNorm2d(mid_ch)
                       if expansion!=1 else nn.Identity())

        self.dw = nn.Conv2d(mid_ch,mid_ch,kernel_size=k,stride=s,
                            padding=k//2,groups=mid_ch,bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.se = SEBlock(mid_ch, se_ratio)

        self.project = nn.Conv2d(mid_ch,out_ch,1,bias=False)
        self.bn2     = nn.BatchNorm2d(out_ch)

        self.drop_connect = drop_connect

    def forward(self, x):
        out = swish(self.bn0(self.expand(x)))
        out = swish(self.bn1(self.dw(out)))
        out = self.se(out)
        out = self.bn2(self.project(out))
        if self.has_residual and self.training and self.drop_connect>0:
            keep = 1 - self.drop_connect
            mask = torch.rand(x.size(0),1,1,1,device=x.device) < keep
            out = out.div(keep) * mask
            out = out + x
        return out
