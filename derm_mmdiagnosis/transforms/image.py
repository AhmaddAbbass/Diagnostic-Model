from torchvision import transforms

# Contrastive augmentations
contrast_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8,0.8,0.8,0.2),
    transforms.RandomGrayscale(0.2),
    transforms.GaussianBlur(23),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])

# Eval / linear‑head transforms
eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])
