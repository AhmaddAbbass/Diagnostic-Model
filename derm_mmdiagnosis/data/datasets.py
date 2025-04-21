from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class TwoViewImageFolder(Dataset):
    """Return two independent augmentations per image (for SimCLR)."""
    def __init__(self, root, transform):
        self.folder = ImageFolder(root, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        img, _ = self.folder[idx]
        return self.transform(img), self.transform(img)

class ClassificationDataset(Dataset):
    """Single‑view ImageFolder for linear evaluation."""
    def __init__(self, root, transform):
        self.folder = ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        return self.folder[idx]    # (img, label)
import torch
from torch.utils.data import Dataset
import pandas as pd

class TextClassificationDataset(Dataset):
    """
    Loads a text‐only metadata file (Excel/CSV) with columns ['text','label_idx'].
    Tokenization is done outside via transforms.text.get_tokenizer().
    """
    def __init__(self, meta_path, tokenizer):
        self.df = pd.read_excel(meta_path)  # or read_csv
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        label = int(row["label_idx"])
        toks = self.tokenizer(text)  # returns {'input_ids':..., 'attention_mask':...}
        return {
            "input_ids":   toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "labels":      torch.tensor(label, dtype=torch.long)
        }
