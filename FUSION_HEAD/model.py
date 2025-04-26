import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel
#@title 🧩 EfficientNet-B0 + SimCLR wrapper & FP16-safe InfoNCE { display-mode: "code" }
import torch
from torchvision import models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes:int=0, pretrained:bool=True):
        super().__init__()
        b0 = models.efficientnet_b0(
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        self.features, self.avgpool = b0.features, b0.avgpool
        in_feats = b0.classifier[1].in_features
        self.classifier = (nn.Linear(in_feats, num_classes)
                           if num_classes else nn.Identity())
        self.num_classes = num_classes

    def extract_features(self, x):
        return torch.flatten(self.avgpool(self.features(x)), 1)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classifier(feats) if self.num_classes else feats

class SimCLRModel(nn.Module):
    def __init__(self, proj_dim:int=128, num_classes:int=0):
        super().__init__()
        self.backbone  = EfficientNetB0(num_classes, pretrained=True)
        self.projector = nn.Sequential(
            nn.Linear(1280,1280,bias=False),
            nn.BatchNorm1d(1280), nn.ReLU(inplace=True),
            nn.Linear(1280,proj_dim)
        )

    def forward(self, x, *, return_logits=False):
        v = self.backbone.extract_features(x)
        z = F.normalize(self.projector(v), dim=1)
        lg= self.backbone.classifier(v) if (return_logits and self.backbone.num_classes) else None
        return v, z, lg
class TextEncoder(nn.Module):
    def __init__(self,num_classes:int, base_model:str=TOKENIZER_NAME,
                 from_scratch:bool=True):
        super().__init__()
        if from_scratch:
            cfg      = AutoConfig.from_pretrained(base_model)
            self.bert= AutoModel.from_config(cfg)   # 💥 random weights
        else:
            self.bert= AutoModel.from_pretrained(base_model)
        hid = self.bert.config.hidden_size
        self.cls_head = nn.Linear(hid,num_classes)
    def forward(self,input_ids,attention_mask,labels=None):
        out   = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask,
                          return_dict=True)
        cls   = out.last_hidden_state[:,0]      # [CLS]
        logits= self.cls_head(cls)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits,labels)
            return logits,loss
        return logits

class FusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.act = nn.ReLU(True)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, u):
        return self.fc2(self.drop(self.act(self.fc1(u))))