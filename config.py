import torch
import os

# Hyperparameters
z_dim = 100
num_classes = 102
image_size = 64
batch_size = 64
lr_G = 0.0002
lr_D = 0.0001
epochs = 50

# Paths
sample_dir = os.path.join("outputs", "samples")
os.makedirs(sample_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
