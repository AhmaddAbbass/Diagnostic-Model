import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from text_encoder import TextEncoder
from dataset import TextDS  # Assuming you have a custom dataset loader

# Hyperparameters
EPOCHS = 20
BATCH_TRAIN = 16
BATCH_VAL = 32
LR = 1e-4
MAX_LEN = 256
WARMUP_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
train_df = load_split('train')  # Assuming `load_split` is a function from earlier
val_df = load_split('val')
cls2id = {c:i for i,c in enumerate(sorted(train_df.label.unique()))}

train_ds = TextDS(train_df, cls2id, MAX_LEN)
val_ds = TextDS(val_df, cls2id, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False)

# Initialize model
model = TextEncoder(len(cls2id), base_model="emilyalsentzer/Bio_ClinicalBERT", from_scratch=False).to(DEVICE)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()  # Set model to training mode
    train_loss = 0

    for batch in train_dl:
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()

        # Forward pass
        logits, loss = model(**b)
        loss.backward()

        # Optimizer step
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    # Print training loss for the epoch
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss / len(train_dl)}")

    # Validation step
    model.eval()  # Set model to evaluation mode
    correct = total = 0

    with torch.no_grad():
        for batch in val_dl:
            b = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**b)[0]
            preds = logits.argmax(1)

            correct += (preds == b["labels"]).sum().item()
            total += preds.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch {epoch}/{EPOCHS} | Validation Accuracy: {val_acc:.2f}%")
