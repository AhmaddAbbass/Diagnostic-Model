import torch
from torch.utils.data import DataLoader
from text_encoder import TextEncoder
from dataset import TextDS  # Assuming you have a custom dataset loader

# Hyperparameters
BATCH_VAL = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load test dataset
test_df = load_split('test')  # Assuming `load_split` is a function from earlier
cls2id = {c:i for i,c in enumerate(sorted(test_df.label.unique()))}
test_ds = TextDS(test_df, cls2id, MAX_LEN)

test_dl = DataLoader(test_ds, batch_size=BATCH_VAL, shuffle=False)

# Initialize model
model = TextEncoder(len(cls2id), base_model="emilyalsentzer/Bio_ClinicalBERT", from_scratch=False).to(DEVICE)

# Load model checkpoint
model_checkpoint = "/path/to/best_checkpoint"  # Update with your checkpoint path
model.load_state_dict(torch.load(model_checkpoint))
model.eval()  # Set model to evaluation mode

correct = total = 0

with torch.no_grad():
    for batch in test_dl:
        b = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**b)[0]
        preds = logits.argmax(1)

        correct += (preds == b["labels"]).sum().item()
        total += preds.size(0)

# Print test accuracy
test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")
