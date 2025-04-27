import torch
from pathlib import Path

def collate(batch):
    imgs = torch.stack([b["pixel"] for b in batch])
    txts = [b["text"] for b in batch]
    labs = torch.tensor([b["label"] for b in batch])
    hi = torch.tensor([b["has_img"] for b in batch])
    ht = torch.tensor([b["has_txt"] for b in batch])
    enc = TOK(txts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return {"images": imgs, "input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labs, "has_img": hi, "has_txt": ht}

def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def test_accuracy(test_dl, model):
    correct = tot = 0
    with torch.no_grad():
        for b in test_dl:
            z = model(b["images"].to(DEVICE))
            pred = z.argmax(1)
            correct += (pred == b["labels"].to(DEVICE)).sum().item()
            tot += pred.size(0)
    print(f"Test accuracy: {100 * correct / tot:.2f}% ({correct}/{tot})")
