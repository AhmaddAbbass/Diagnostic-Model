# utils/transforms.py
"""
transforms.py
-------------
(Section “End‑to‑End Pipeline” – Preprocessing)
Holds all image transform pipelines:
  • contrast_transforms for SimCLR,
  • eval_transforms for linear/fusion/evaluation.
Shared by all training scripts.
"""




from torchvision import transforms

contrast_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(0.2),
    transforms.GaussianBlur(23),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
