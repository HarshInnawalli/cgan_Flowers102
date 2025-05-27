import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from config import num_classes

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels + 1, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final = spectral_norm(nn.Conv2d(512, 1, 4, 1, 0))

    def forward(self, img, labels):
        batch_size, _, h, w = img.size()
        label_map = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).float()
        label_map = label_map.repeat(1, 1, h, w) / num_classes
        d_in = torch.cat((img, label_map), dim=1)
        features = self.model(d_in)
        validity = torch.sigmoid(self.final(features)).view(-1)
        return validity, features.mean([2, 3])
