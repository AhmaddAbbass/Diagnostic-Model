from model import EffNet, FusionMLP, TextEncoder
from train import train_model
from test import test_accuracy

# Initialize and train
train_model()

# Test
test_accuracy(test_dl, fusion)
