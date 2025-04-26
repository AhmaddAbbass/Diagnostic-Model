import torch
from model import FusionMLP
from utils import test_accuracy

# Load the best model and test
fusion = FusionMLP().to(DEVICE)
fusion.load_state_dict(torch.load("best_mlp.pt", map_location=DEVICE))
fusion.eval()
test_accuracy(test_dl, fusion)
