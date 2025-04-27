import torch
from transformers import AutoTokenizer

# make sure MAX_LEN and TOK are imported or defined here
from dataset import MAX_LEN, TOK

def collate(batch):
    imgs = torch.stack([b["pixel"] for b in batch])
    txts = [b["text"] for b in batch]
    labs = torch.tensor([b["label"] for b in batch])
    hi   = torch.tensor([b["has_img"] for b in batch])
    ht   = torch.tensor([b["has_txt"] for b in batch])

    enc = TOK(
        txts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return {
        "images":         imgs,
        "input_ids":      enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels":         labs,
        "has_img":        hi,
        "has_txt":        ht
    }

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def test_accuracy(dl, model, DEVICE="cpu"):
    correct = tot = 0
    model.eval()
    with torch.no_grad():
        for b in dl:
            zi = model(b["images"].to(DEVICE))
            pred = zi.argmax(1)
            correct += (pred == b["labels"].to(DEVICE)).sum().item()
            tot     += pred.size(0)
    print(f"Test acc: {100*correct/tot:.2f}%")
